# =============================================================================
# Le moteur de l'application : construire les variables, entraîner le meilleur
# modèle (V4, machine learning) et prédire le risque de signalement.
# Même logique que 2_machine_learning.py, rangée en fonctions pour Streamlit.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 3
VARIABLES = ["ventes_mois", "ventes_3mois", "ventes_instables", "volume", "signal_3mois", "signal_12mois",
             "signal_24mois", "classe_signalee", "mitm", "un_seul_labo", "injectable"]


def charger(dossier):
    """Lit les 3 fichiers propres (sortis de 1_nettoyage_final.py) et garde les médicaments suivables."""
    ventes = pd.read_csv(f"{dossier}/ventes_par_medicament.csv", dtype={"cis": str})
    signal = pd.read_csv(f"{dossier}/signalements_propres.csv", dtype={"cis": str, "cip13": str}, parse_dates=["date"])
    fiches = pd.read_csv(f"{dossier}/fiches_medicaments.csv", dtype={"cis": str})
    volume = ventes.groupby("cis").boites.median()
    fiches = fiches[fiches.cis.isin(volume[volume >= 100].index) & ~fiches.arrete & ~fiches.vaccin].copy()
    ventes = ventes[ventes.cis.isin(fiches.cis)]
    return ventes, signal, fiches


def construire(ventes, signal, fiches):
    """Une ligne par médicament et par mois. 'cible' est vide quand l'avenir n'est pas encore connu."""
    tab = ventes.pivot(index="cis", columns="mois", values="unites").fillna(0)
    mois = list(tab.columns)
    signal = signal.copy()
    signal["mois"] = signal.date.dt.strftime("%Y-%m")
    dernier_signal = signal.mois.max()                         # les listes ANSM s'arrêtent ici
    sig = pd.crosstab(signal.cis, signal.mois).clip(upper=1)
    tous = sorted(set(sig.columns) | set(mois))
    sig = sig.reindex(index=tab.index, columns=tous, fill_value=0)

    f = fiches.copy()
    labos = f.groupby("traitement").titulaire.nunique()
    f["un_seul_labo"] = (f.traitement.map(labos) == 1).astype(int)
    f["classe"] = f.atc.str[:4]
    f = f.set_index("cis").reindex(tab.index)

    lignes = []
    for t in range(6, len(mois)):
        m = mois[t]
        j = tous.index(m)
        passe12 = tab.iloc[:, max(0, t - 12):t]
        normal = passe12.median(axis=1).replace(0, np.nan)
        d = pd.DataFrame(index=tab.index)
        d["mois"] = m
        d["ventes_mois"] = (tab.iloc[:, t] / normal).clip(upper=2)
        d["ventes_3mois"] = (tab.iloc[:, t - 2:t + 1].mean(axis=1) / normal).clip(upper=2)
        d["ventes_instables"] = (passe12.std(axis=1) / passe12.mean(axis=1)).clip(upper=3)
        d["volume"] = np.log10(passe12.mean(axis=1) + 1)
        d["signal_3mois"] = sig.iloc[:, max(0, j - 2):j + 1].sum(axis=1)
        d["signal_12mois"] = sig.iloc[:, max(0, j - 11):j + 1].sum(axis=1)
        d["signal_24mois"] = sig.iloc[:, max(0, j - 23):j + 1].sum(axis=1)
        d["classe_signalee"] = (d.signal_3mois > 0).groupby(f.classe).transform("mean")
        d["mitm"] = f.mitm.astype(int)
        d["un_seul_labo"] = f.un_seul_labo
        d["injectable"] = (f.fam == "inj").astype(int)
        connu = j + HORIZON < len(tous) and tous[j + HORIZON] <= dernier_signal
        d["cible"] = (sig.iloc[:, j + 1:j + 1 + HORIZON].sum(axis=1) > 0).astype(float) if connu else np.nan
        d["historique_complet"] = m <= dernier_signal         # signalements connus jusqu'à ce mois ?
        lignes.append(d.reset_index())
    data = pd.concat(lignes, ignore_index=True)
    data[VARIABLES] = data[VARIABLES].fillna(0)
    return data, dernier_signal


def modele_v4():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=50, random_state=0)


def modele_v3():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def entrainer(data):
    """Le modèle final apprend sur TOUS les mois dont on connaît la réponse."""
    appr = data.dropna(subset=["cible"])
    return modele_v4().fit(appr[VARIABLES], appr.cible), appr.cible.mean()


def evaluer(data):
    """Le même test qu'au script 2 : apprendre jusqu'à 2025-05, tester sur 2025-08 et 2025-09."""
    appr = data[data.mois <= "2025-05"].dropna(subset=["cible"])
    test = data[data.mois.between("2025-08", "2025-09")].dropna(subset=["cible"])
    res = {}
    scores = {"V1 répéter le passé": test.signal_12mois,
              "V2 points à la main": (3 * (test.signal_3mois > 0) + 2 * (test.signal_12mois > 0)
                                      + (test.signal_24mois >= 3) + (test.ventes_3mois < 0.85)
                                      + (test.classe_signalee >= 0.1) + test.un_seul_labo),
              "V3 points appris": modele_v3().fit(appr[VARIABLES], appr.cible).predict_proba(test[VARIABLES])[:, 1],
              "V4 machine learning": modele_v4().fit(appr[VARIABLES], appr.cible).predict_proba(test[VARIABLES])[:, 1]}
    for nom, s in scores.items():
        s = pd.Series(np.asarray(s, dtype=float), index=test.index)
        top = [test.cible[s[test.mois == m].nlargest(100).index].mean() for m in test.mois.unique()]
        res[nom] = {"AUC": roc_auc_score(test.cible, s), "AP": average_precision_score(test.cible, s),
                    "Top 100": float(np.mean(top))}
    return pd.DataFrame(res).T, len(test), test.cible.mean()


def raisons(r):
    """Les signes d'alerte lisibles par un humain, à partir des variables d'une ligne."""
    out = []
    if r.signal_3mois > 0:
        out.append("signalé ces 3 derniers mois")
    elif r.signal_12mois > 0:
        out.append("signalé dans l'année")
    elif r.signal_24mois > 0:
        out.append("signalé ces 2 dernières années")
    if r.ventes_3mois < 0.85:
        out.append(f"ventes à {r.ventes_3mois:.0%} de la normale")
    if r.classe_signalee >= 0.10:
        out.append(f"famille tendue ({r.classe_signalee:.0%} signalés)")
    if r.mitm:
        out.append("intérêt thérapeutique majeur")
    if r.un_seul_labo:
        out.append("un seul laboratoire")
    return " · ".join(out) or "aucun signe particulier"
