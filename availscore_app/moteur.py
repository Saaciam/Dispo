# =============================================================================
# Le moteur de l'application.
#
# Question du pharmacien : « ce médicament est-il en rupture, et risque-t-il de l'être
# pendant tel mois ? »
#   1. Situation actuelle : le statut publié aujourd'hui par l'ANSM dans la BDPM
#      (rupture de stock, tension d'approvisionnement, remise à disposition).
#   2. Prévision pour un mois choisi : la probabilité que le laboratoire déclare à l'ANSM
#      une rupture de stock ou un risque de rupture pour ce médicament PENDANT ce mois,
#      puis la part estimée de chaque type (rupture / risque de rupture).
#
# Prédire à partir d'aujourd'hui : les ventes Medic'AM ont quelques mois de retard, les
# listes ANSM bien plus. Pour chaque mois proposé, un modèle est entraîné dans la MÊME
# situation : déclarations vues avec le même retard, et même distance entre les dernières
# ventes connues et le mois prédit. Tout est recalculé avec la date du jour.
# =============================================================================

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NB_MOIS = 6              # le mois en cours et les 5 suivants
VARIABLES = ["ventes_mois", "ventes_3mois", "ventes_instables", "volume", "tendance_6mois",
             "signal_3mois", "signal_12mois", "signal_24mois", "nb_declarations_12mois", "mois_depuis_declaration",
             "classe_signalee", "equivalents_declares", "ventes_traitement_3mois", "part_marche",
             "mitm", "un_seul_labo", "n_labos", "n_equivalents", "injectable"]
NOMS_VARIABLES = {
    "ventes_mois": "Ventes du mois / mois normal", "ventes_3mois": "Ventes des 3 derniers mois / normal",
    "ventes_instables": "Ventes irrégulières", "volume": "Volume de ventes",
    "tendance_6mois": "Tendance des ventes sur 6 mois",
    "signal_3mois": "Déclaré sur 3 mois (derniers connus)", "signal_12mois": "Déclaré sur 12 mois (derniers connus)",
    "signal_24mois": "Déclaré sur 24 mois (derniers connus)",
    "nb_declarations_12mois": "Nombre de déclarations sur 12 mois", "mois_depuis_declaration": "Mois depuis la dernière déclaration",
    "classe_signalee": "Famille ATC tendue", "equivalents_declares": "Équivalents déclarés",
    "ventes_traitement_3mois": "Ventes de tous les équivalents / normal", "part_marche": "Part des ventes parmi les équivalents",
    "mitm": "Intérêt thérapeutique majeur (MITM)", "un_seul_labo": "Un seul laboratoire",
    "n_labos": "Nombre de laboratoires", "n_equivalents": "Nombre d'équivalents", "injectable": "Injectable"}
GRAVITE = {"Rupture de stock": 0, "Tension d'approvisionnement": 1, "Arrêt de commercialisation": 2,
           "Remise à disposition": 3}


def rien(fraction, message):
    pass


def mois_entre(a, b):
    """Nombre de mois de a à b ('2025-12', '2026-08' -> 8)."""
    return (pd.Period(b, "M") - pd.Period(a, "M")).n


def plus_mois(m, n):
    return str(pd.Period(m, "M") + n)


# ------------------------------------------------------------------ lecture
def charger(dossier):
    """Lit les 3 fichiers propres (sortis de 1_nettoyage_final.py) et garde les médicaments suivables :
    au moins 100 boîtes par mois en médiane, encore commercialisés, hors vaccins."""
    ventes = pd.read_csv(f"{dossier}/ventes_par_medicament.csv", dtype={"cis": str})
    signal = pd.read_csv(f"{dossier}/signalements_propres.csv", dtype={"cis": str, "cip13": str}, parse_dates=["date"])
    tout = pd.read_csv(f"{dossier}/fiches_medicaments.csv", dtype={"cis": str})
    volume = ventes.groupby("cis").boites.median()
    fiches = tout[tout.cis.isin(volume[volume >= 100].index) & ~tout.arrete & ~tout.vaccin].copy()
    ventes = ventes[ventes.cis.isin(fiches.cis)]
    return ventes, signal, fiches, tout


def charger_disponibilite(dossier):
    """Le statut ACTUEL publié par l'ANSM dans la BDPM. Un statut par médicament (le plus grave)."""
    chemin = f"{dossier}/disponibilite_ansm.csv"
    if not os.path.exists(chemin):
        return pd.DataFrame(columns=["cis", "statut", "date_debut", "date_maj", "date_remise", "lien"])
    d = pd.read_csv(chemin, dtype=str)
    d["ordre"] = d.statut.map(GRAVITE).fillna(9)
    d["maj"] = pd.to_datetime(d.date_maj, format="%d/%m/%Y", errors="coerce")
    d = d.sort_values(["cis", "ordre", "maj"], ascending=[True, True, False]).drop_duplicates("cis")
    return d[["cis", "statut", "date_debut", "date_maj", "date_remise", "lien"]]


def charger_types(dossier, k=10):
    """Part des ruptures (par opposition aux risques de rupture) dans les déclarations, d'après le
    fichier Data.ansm « historique ruptures ». Pour chaque médicament : son propre historique,
    rapproché de la part nationale de la dernière année (k = poids de la part nationale)."""
    chemin = f"{dossier}/historique_types_ansm.csv"
    if not os.path.exists(chemin):
        return pd.DataFrame(columns=["cis", "n_ruptures", "n_risques", "part_rupture"]), np.nan, None
    h = pd.read_csv(chemin, dtype={"cis": str}, parse_dates=["date"])
    if h.empty:
        return pd.DataFrame(columns=["cis", "n_ruptures", "n_risques", "part_rupture"]), np.nan, None
    h["rupture"] = (h.classification == "rupture").astype(int)
    annee = h.date.dt.year.max()
    p0 = h[h.date.dt.year == annee].rupture.mean()
    g = h.groupby("cis").rupture.agg(n_ruptures="sum", n="size").reset_index()
    g["n_risques"] = g.n - g.n_ruptures
    g["part_rupture"] = (g.n_ruptures + k * p0) / (g.n + k)
    periode = (h.date.min().strftime("%m/%Y"), h.date.max().strftime("%m/%Y"))
    return g[["cis", "n_ruptures", "n_risques", "part_rupture"]], p0, periode


# ------------------------------------------------------------------ situation et variables
def situation(ventes, signal, aujourd_hui=None):
    """Où en sont les données par rapport à aujourd'hui."""
    dernier_vente = ventes.mois.max()
    dernier_signal = signal.date.max().strftime("%Y-%m")
    maintenant = str(pd.Period(aujourd_hui or pd.Timestamp.today(), "M"))
    proposes = [plus_mois(maintenant, i) for i in range(NB_MOIS)]
    return {"maintenant": maintenant, "dernier_vente": dernier_vente, "dernier_signal": dernier_signal,
            "retard": max(0, mois_entre(dernier_signal, dernier_vente)), "mois": proposes,
            "ecarts": {m: max(1, mois_entre(dernier_vente, m)) for m in proposes}}


def construire(ventes, signal, fiches, retard, ecarts, avance=rien):
    """Une ligne par médicament et par mois t, avec ce qu'on aurait su à la fin du mois t si les
    déclarations n'étaient connues que jusqu'à t - retard. Pour chaque écart h : 'cible_h' = au moins
    une déclaration pendant le mois t + h (vide si on ne le sait pas encore)."""
    tab = ventes.pivot(index="cis", columns="mois", values="unites").fillna(0)
    mois = list(tab.columns)
    signal = signal.copy()
    signal["mois"] = signal.date.dt.strftime("%Y-%m")
    dernier_signal = signal.mois.max()
    nb = pd.crosstab(signal.cis, signal.mois)                 # nombre de déclarations (une par présentation)
    cal = [str(p) for p in pd.period_range(min(nb.columns.min(), mois[0]),
                                           pd.Period(mois[-1], "M") + max(ecarts), freq="M")]
    pos = {m: i for i, m in enumerate(cal)}
    nb = nb.reindex(index=tab.index, columns=cal, fill_value=0)
    sig = nb.clip(upper=1)

    f = fiches.set_index("cis").reindex(tab.index)
    f["classe"] = f.atc.str[:4]
    f["n_labos"] = f.traitement.map(fiches.groupby("traitement").titulaire.nunique())
    f["n_equivalents"] = f.traitement.map(fiches.groupby("traitement").size())
    ventes_trait = tab.groupby(f.traitement).transform("sum")

    lignes = []
    for t in range(6, len(mois)):                              # au moins 6 mois de ventes passées
        m = mois[t]
        avance((t - 6) / max(1, len(mois) - 6), f"Calcul des indicateurs du mois {m}")
        j = pos[m] - retard                                    # dernier mois de déclarations visible
        if j < 0 or cal[j] > dernier_signal:
            continue
        p12 = tab.iloc[:, max(0, t - 12):t]
        normal = p12.median(axis=1).replace(0, np.nan)
        d = pd.DataFrame(index=tab.index)
        d["mois"] = m
        d["ventes_mois"] = (tab.iloc[:, t] / normal).clip(upper=2)
        d["ventes_3mois"] = (tab.iloc[:, t - 2:t + 1].mean(axis=1) / normal).clip(upper=2)
        d["ventes_instables"] = (p12.std(axis=1) / p12.mean(axis=1)).clip(upper=3)
        d["volume"] = np.log10(p12.mean(axis=1) + 1)
        d["tendance_6mois"] = ((tab.iloc[:, t - 2:t + 1].mean(axis=1) - tab.iloc[:, t - 5:t - 2].mean(axis=1))
                               / normal).clip(-2, 2)
        d["signal_3mois"] = sig.iloc[:, max(0, j - 2):j + 1].sum(axis=1)
        d["signal_12mois"] = sig.iloc[:, max(0, j - 11):j + 1].sum(axis=1)
        d["signal_24mois"] = sig.iloc[:, max(0, j - 23):j + 1].sum(axis=1)
        d["nb_declarations_12mois"] = np.log1p(nb.iloc[:, max(0, j - 11):j + 1].sum(axis=1))
        vu = sig.iloc[:, :j + 1].values
        depuis = vu[:, ::-1].argmax(axis=1).astype(float) + retard
        depuis[vu.sum(axis=1) == 0] = 60
        d["mois_depuis_declaration"] = np.minimum(depuis, 60)
        d["classe_signalee"] = (d.signal_3mois > 0).groupby(f.classe).transform("mean")
        d["equivalents_declares"] = (d.signal_3mois > 0).groupby(f.traitement).transform("mean")
        nt = ventes_trait.iloc[:, max(0, t - 12):t].median(axis=1).replace(0, np.nan)
        d["ventes_traitement_3mois"] = (ventes_trait.iloc[:, t - 2:t + 1].mean(axis=1) / nt).clip(upper=2)
        d["part_marche"] = (tab.iloc[:, t - 2:t + 1].mean(axis=1)
                            / ventes_trait.iloc[:, t - 2:t + 1].mean(axis=1).replace(0, np.nan))
        d["mitm"] = f.mitm.astype(int)
        d["un_seul_labo"] = (f.n_labos == 1).astype(int)
        d["n_labos"] = f.n_labos.clip(upper=10)
        d["n_equivalents"] = f.n_equivalents.clip(upper=30)
        d["injectable"] = (f.fam == "inj").astype(int)
        for h in sorted(set(ecarts)):
            k = pos[m] + h
            d[f"cible_{h}"] = sig.iloc[:, k].astype(float) if cal[k] <= dernier_signal else np.nan
        lignes.append(d.reset_index())
    data = pd.concat(lignes, ignore_index=True)
    data[VARIABLES] = data[VARIABLES].fillna(0)
    return data


def modele_v4():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=100,
                                          l2_regularization=1.0, random_state=0)


def modele_v3():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def v2_points(x):
    return (3 * (x.signal_3mois > 0) + 2 * (x.signal_12mois > 0) + (x.signal_24mois >= 3)
            + (x.ventes_3mois < 0.85) + (x.classe_signalee >= 0.1) + x.un_seul_labo)


def niveau(risque, moyenne):
    """Élevé = au moins 3 fois la moyenne du mois ; À surveiller = au moins 1,5 fois."""
    return pd.cut(np.asarray(risque) / moyenne, [-1, 1.5, 3, np.inf],
                  labels=["Faible", "À surveiller", "Élevé"], right=False).astype(str)


# ------------------------------------------------------------------ prédiction
def preparer(dossier, avance=rien, aujourd_hui=None, cache=None):
    """Prédit, pour chaque médicament suivi et chaque mois proposé, la probabilité d'une déclaration
    de rupture ou de risque de rupture pendant ce mois. Résultat gardé sur disque (cache) tant que
    les données et le mois en cours ne changent pas."""
    avance(0.01, "Lecture des fichiers propres")
    ventes, signal, fiches, tout = charger(dossier)
    S = situation(ventes, signal, aujourd_hui)
    cle = (S["maintenant"], tuple(os.path.getmtime(f"{dossier}/{f}") for f in
                                  ["ventes_par_medicament.csv", "signalements_propres.csv", "fiches_medicaments.csv"]))
    if cache and os.path.exists(cache):
        try:
            with open(cache, "rb") as fh:
                sauve = pickle.load(fh)
            if sauve["cle"] == cle:
                avance(1, "Prédiction relue")
                return completer(sauve["P"], dossier, ventes, signal, tout)
        except Exception:
            pass

    ecarts = sorted(set(S["ecarts"].values()))
    avance(0.03, f"Construction du tableau médicament × mois (déclarations vues avec {S['retard']} mois de retard)")
    data = construire(ventes, signal, fiches, S["retard"], ecarts, lambda f, m: avance(0.03 + 0.37 * f, m))
    dernier = data[data.mois == S["dernier_vente"]].copy()

    modeles, moyennes, n_appr = {}, {}, {}
    for i, h in enumerate(ecarts):
        mois_vise = [m for m in S["mois"] if S["ecarts"][m] == h][0]
        avance(0.40 + 0.55 * i / len(ecarts), f"Entraînement du modèle pour {pd.Period(mois_vise, 'M').strftime('%m/%Y')}"
                                               f" ({h} mois après les dernières ventes)")
        appr = data.dropna(subset=[f"cible_{h}"])
        if appr[f"cible_{h}"].nunique() < 2:
            raise ValueError("Pas assez de recul pour apprendre : ajoutez des listes ANSM plus récentes.")
        modeles[h] = modele_v4().fit(appr[VARIABLES], appr[f"cible_{h}"])
        moyennes[h] = appr[f"cible_{h}"].mean()
        n_appr[h] = (appr.mois.min(), appr.mois.max(), appr.mois.nunique())
        dernier[f"risque_{h}"] = modeles[h].predict_proba(dernier[VARIABLES])[:, 1]

    P = {"situation": S, "modeles": modeles, "moyennes": moyennes, "n_appr": n_appr,
         "actuel": dernier.drop(columns=[c for c in dernier if c.startswith("cible_")])}
    if cache:
        try:
            with open(cache, "wb") as fh:
                pickle.dump({"cle": cle, "P": P}, fh)
        except OSError:
            pass
    avance(1, "Prédiction terminée")
    return completer(P, dossier, ventes, signal, tout)


def completer(P, dossier, ventes, signal, tout):
    """Ajoute ce qui ne demande pas de calcul : fiches, statut actuel, part des ruptures."""
    P = dict(P)
    a = P["actuel"].merge(tout[["cis", "nom", "titulaire", "atc", "traitement"]], on="cis")
    types, p0, periode = charger_types(dossier)
    a = a.merge(types, on="cis", how="left")
    a["part_rupture"] = a.part_rupture.fillna(p0)
    P.update(actuel=a, ventes=ventes, signal=signal, tout=tout, dispo=charger_disponibilite(dossier),
             part_rupture_nationale=p0, periode_types=periode, types=types)
    return P


def pour_le_mois(P, mois):
    """Le tableau des médicaments pour un mois proposé, trié du plus au moins à risque."""
    h = P["situation"]["ecarts"][mois]
    a = P["actuel"].copy()
    a["risque"] = a[f"risque_{h}"]
    a["moyenne"] = P["moyennes"][h]
    a["fois_moyenne"] = a.risque / a.moyenne
    a["niveau"] = niveau(a.risque, P["moyennes"][h])
    a["risque_rupture"] = a.risque * a.part_rupture
    a["risque_tension"] = a.risque * (1 - a.part_rupture)
    a = a.merge(P["dispo"], on="cis", how="left").sort_values("risque", ascending=False).reset_index(drop=True)
    a["rang"] = range(1, len(a) + 1)
    return a


# ------------------------------------------------------------------ performance
def top_k(mois_test, y, score, k=100):
    """Parmi les k médicaments les plus à risque de chaque mois, part vraiment déclarée."""
    s = pd.Series(np.asarray(score, dtype=float))
    y = np.asarray(y)
    mt = np.asarray(mois_test)
    return float(np.mean([y[mt == m][np.argsort(-s[mt == m].values)[:k]].mean() for m in np.unique(mt)]))


def evaluer(dossier, P, avance=rien):
    """Note les modèles dans la situation réelle de chaque mois proposé (même retard, même écart),
    sur les 2 derniers mois dont on connaît la réponse, jamais vus pendant l'apprentissage."""
    S = P["situation"]
    ventes, signal, fiches, _ = charger(dossier)
    ecarts = sorted(set(S["ecarts"].values()))
    avance(0.02, "Construction du tableau pour le test")
    data = construire(ventes, signal, fiches, S["retard"], ecarts, lambda f, m: avance(0.02 + 0.28 * f, m))

    def decoupe(h):
        c = f"cible_{h}"
        d = data.dropna(subset=[c])
        connus = sorted(d.mois.unique())
        test_mois = connus[-2:]
        fin = plus_mois(test_mois[0], -h)
        return d[d.mois <= fin], d[d.mois.isin(test_mois)], fin, test_mois

    # 1. Une ligne par mois proposé. On garde le jeu de test et les modèles entraînés SANS les mois
    #    du test : la page « Évaluation » s'en sert pour tester des exemples un par un.
    par_mois, tests = [], {}
    for i, m in enumerate(S["mois"]):
        h = S["ecarts"][m]
        avance(0.30 + 0.30 * i / len(S["mois"]), f"Test du modèle pour {pd.Period(m, 'M').strftime('%m/%Y')}")
        a, t, fin, tm = decoupe(h)
        c = f"cible_{h}"
        v3 = modele_v3().fit(a[VARIABLES], a[c])
        v4 = modele_v4().fit(a[VARIABLES], a[c])
        s = v4.predict_proba(t[VARIABLES])[:, 1]
        y = t[c]
        par_mois.append({"Mois prédit": m, "Écart (mois)": h, "AUC": roc_auc_score(y, s),
                         "AP": average_precision_score(y, s), "Top 100": top_k(t.mois, y, s),
                         "Part déclarée": y.mean(), "Brier": brier_score_loss(y, s),
                         "Mois d'apprentissage": a.mois.nunique()})
        jeu = t[["cis", "mois"] + VARIABLES].assign(cible=y.values).reset_index(drop=True)
        jeu["mois_predit"] = jeu.mois.map(lambda x: plus_mois(x, h))
        jeu["v1"] = jeu.signal_12mois
        jeu["v2"] = v2_points(jeu)
        jeu["v3"] = v3.predict_proba(jeu[VARIABLES])[:, 1]
        jeu["v4"] = s
        tests[h] = {"jeu": jeu, "v3": v3, "v4": v4, "moyenne": a[c].mean(), "fin_appr": fin, "test_mois": tm}
    par_mois = pd.DataFrame(par_mois)

    # 2. Détail pour le mois en cours
    h = S["ecarts"][S["maintenant"]]
    a, t, fin, test_mois = decoupe(h)
    c = f"cible_{h}"
    avance(0.62, "Comparaison des 4 versions pour le mois en cours")
    v3, v4 = tests[h]["v3"], tests[h]["v4"]
    scores = {"V1 répéter le passé": t.signal_12mois.values, "V2 points à la main": v2_points(t).values,
              "V3 points appris": v3.predict_proba(t[VARIABLES])[:, 1],
              "V4 machine learning": v4.predict_proba(t[VARIABLES])[:, 1]}
    neufs = (t.signal_12mois == 0).values
    notes = pd.DataFrame({nom: {"AUC": roc_auc_score(t[c], s), "AP": average_precision_score(t[c], s),
                                "Top 100": top_k(t.mois, t[c], s),
                                "AUC jamais déclarés": (roc_auc_score(t[c][neufs], s[neufs])
                                                        if nom != "V1 répéter le passé" else np.nan)}
                          for nom, s in scores.items()}).T

    avance(0.70, "Précision selon la longueur de la liste")
    ks = [10, 25, 50, 100, 200, 300, 500, 1000]
    courbe = pd.DataFrame([{"Longueur de la liste": k, "Version": nom, "Part déclarée": top_k(t.mois, t[c], s, k)}
                           for nom, s in scores.items() for k in ks])
    courbe = pd.concat([courbe, pd.DataFrame({"Longueur de la liste": ks, "Version": "Hasard",
                                              "Part déclarée": t[c].mean()})])

    avance(0.75, "Fiabilité des probabilités")
    s4 = scores["V4 machine learning"]
    fiab = (pd.DataFrame({"Niveau": niveau(s4, a[c].mean()), "y": t[c].values, "p": s4})
            .groupby("Niveau").agg(Médicaments=("y", "size"), Risque_prédit=("p", "mean"), Vraiment_déclarés=("y", "mean"))
            .reindex(["Élevé", "À surveiller", "Faible"]))

    avance(0.80, "Importance des variables")
    imp = permutation_importance(v4, t[VARIABLES], t[c], scoring="roc_auc", n_repeats=3, random_state=0)
    importance = (pd.DataFrame({"Variable": [NOMS_VARIABLES[v] for v in VARIABLES], "Baisse de l'AUC": imp.importances_mean})
                  .sort_values("Baisse de l'AUC", ascending=False))

    # 3. Rupture ou risque de rupture : l'historique du médicament le prédit-il ?
    avance(0.92, "Test de la séparation rupture / risque de rupture")
    types_test = None
    chemin = f"{dossier}/historique_types_ansm.csv"
    if os.path.exists(chemin):
        hh = pd.read_csv(chemin, dtype={"cis": str}, parse_dates=["date"])
        hh["r"] = (hh.classification == "rupture").astype(int)
        derniere = hh.date.dt.year.max()
        av, ap = hh[hh.date.dt.year < derniere], hh[hh.date.dt.year == derniere]
        if len(av) and ap.r.nunique() > 1:
            p0 = av[av.date.dt.year == derniere - 1].r.mean()
            g = av.groupby("cis").r.agg(["sum", "size"])
            q = (ap.cis.map(g["sum"]).fillna(0) + 10 * p0) / (ap.cis.map(g["size"]).fillna(0) + 10)
            types_test = {"annee": int(derniere), "n": len(ap), "AUC": roc_auc_score(ap.r, q),
                          "part_reelle": ap.r.mean(), "part_prevue": q.mean()}
    avance(1, "Performance calculée")
    return {"par_mois": par_mois, "notes": notes, "courbe": courbe, "fiabilite": fiab, "importance": importance,
            "test_mois": test_mois, "fin_appr": fin, "n_test": len(t), "part": t[c].mean(), "ecart": h,
            "types": types_test, "tests": tests}


# ------------------------------------------------------------------ évaluation exemple par exemple
VERSIONS = {"v1": "V1 répéter le passé", "v2": "V2 points à la main", "v3": "V3 points appris",
            "v4": "V4 machine learning"}


def alertes(jeu, moyenne):
    """Quand chaque version « donne l'alerte » : V1 si déclaré sur les 12 derniers mois connus,
    V2 à partir de 3 points, V3 et V4 si la probabilité atteint 1,5 fois la moyenne (niveau « À surveiller »)."""
    return {"v1": jeu.v1 > 0, "v2": jeu.v2 >= 3, "v3": jeu.v3 >= 1.5 * moyenne, "v4": jeu.v4 >= 1.5 * moyenne}


def verdict(alerte, declare):
    if alerte:
        return "Alerte justifiée" if declare else "Fausse alerte"
    return "Déclaration manquée" if declare else "Pas d'alerte, à raison"


def matrice(jeu, moyenne):
    """Pour chaque version, le décompte des 4 verdicts sur tout le jeu de test, et deux taux simples."""
    lignes = []
    for v, a in alertes(jeu, moyenne).items():
        y = jeu.cible.astype(bool)
        vp, fp = int((a & y).sum()), int((a & ~y).sum())
        fn, vn = int((~a & y).sum()), int((~a & ~y).sum())
        lignes.append({"Version": VERSIONS[v], "Alerte justifiée": vp, "Fausse alerte": fp,
                       "Déclaration manquée": fn, "Pas d'alerte, à raison": vn,
                       "Alertes justes (précision)": vp / max(1, vp + fp),
                       "Déclarations repérées (rappel)": vp / max(1, vp + fn)})
    return pd.DataFrame(lignes)


def raisons(r):
    """Les signes d'alerte lisibles par un humain, à partir des variables d'une ligne."""
    out = []
    if r.signal_3mois > 0:
        out.append("déclaré sur les 3 derniers mois connus")
    elif r.signal_12mois > 0:
        out.append("déclaré sur les 12 derniers mois connus")
    elif r.signal_24mois > 0:
        out.append("déclaré sur les 24 derniers mois connus")
    if r.ventes_3mois < 0.85:
        out.append(f"ventes à {r.ventes_3mois:.0%} de la normale")
    if r.equivalents_declares >= 0.3:
        out.append("des équivalents sont déclarés")
    if r.classe_signalee >= 0.10:
        out.append(f"famille tendue ({r.classe_signalee:.0%} déclarés)")
    if r.mitm:
        out.append("intérêt thérapeutique majeur")
    if r.un_seul_labo:
        out.append("un seul laboratoire")
    return " · ".join(out) or "aucun signe particulier"
