"""
Pipeline AvailScore : transforme les donnees brutes et applique la formule.

  donnees/               les fichiers bruts
  donnees_transformees/  le resultat de chaque etape + journal_pipeline.csv
  resultats/             le resultat final

  Boites attendues = niveau habituel x saison x effet du mois
  Disponibilite    = boites delivrees / boites attendues (plafonnee a 100 %)
  Dispo prevue     = 100 % - manque d'aujourd'hui x 0,9^h

Lancer a la main :  python pipeline.py
"""
import glob, json, os, re, sys, time, unicodedata, warnings
import numpy as np, pandas as pd, xlrd
warnings.filterwarnings("ignore")

VOLUME_MIN = 1000        # boites par mois : en dessous, la mesure est trop bruitee
SEUIL_MANQUE = 0.70      # sous 70 % : manque reel
SEUIL_TENDU = 0.85       # sous 85 % : tendu
FACTEUR_MENSUEL = 0.90   # chaque mois, 10 % du manque se resorbe


def sans_accent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


def lire_bdpm(data, nom, colonnes):
    chemin = f"{data}/bdpm/{nom}"
    brut = open(chemin, "rb").read()
    try:
        texte = brut.decode("utf-8")          # certains fichiers sont en UTF-8...
    except UnicodeDecodeError:
        texte = brut.decode("latin-1")        # ...d'autres en Latin-1
    lignes = [l.split("\t") for l in texte.splitlines() if l.strip()]
    n = len(colonnes)
    return pd.DataFrame([(l + [None] * n)[:n] for l in lignes], columns=colonnes)


def lire_medicam(data):
    """Une ligne par presentation (CIP13) et par mois. Cellule vide = rien vendu."""
    lignes, infos = [], {}
    for f in sorted(glob.glob(f"{data}/medicam/*.xls")):
        feuille = xlrd.open_workbook(f, on_demand=True).sheet_by_index(1)     # "tous prescripteurs"
        entete = [str(x).replace("\n", " ") for x in feuille.row_values(0)]
        cols = [(i, h[-7:]) for i, h in enumerate(entete) if "boites" in h.lower()]
        for r in range(1, feuille.nrows):
            v = feuille.row_values(r)
            try:
                cip = str(int(v[0]))
            except (ValueError, TypeError):
                continue                                                      # lignes "Total", "Autres"
            infos[cip] = (v[1], v[5], v[7])
            lignes += [(cip, m, v[i]) for i, m in cols]
    df = pd.DataFrame(lignes, columns=["cip13", "mois", "boites"])
    df["boites"] = pd.to_numeric(df.boites, errors="coerce")
    stats = {"fichiers": len(glob.glob(f"{data}/medicam/*.xls")), "negatifs": int((df.boites < 0).sum())}
    df = df.drop_duplicates(["cip13", "mois"], keep="last")      # un mois livre deux fois : le plus recent gagne
    pv = df.pivot_table(index="cip13", columns="mois", values="boites", aggfunc="sum")
    stats["vides"] = int(pv.isna().sum().sum())
    pv = pv.fillna(0).clip(lower=0)
    ref = pd.DataFrame.from_dict(infos, orient="index", columns=["nom_court", "atc", "atc2"])
    return pv, ref, stats


def lire_signalements(data, cip):
    """Date de declaration et CIS. Avant 2022 il n'y a pas de code : on ne les utilise pas ici."""
    tables = []
    for f in sorted(glob.glob(f"{data}/ansm/*.xlsx")):
        for nom, brut in pd.read_excel(f, sheet_name=None, header=None, dtype=str).items():
            for i in range(min(30, len(brut))):
                ligne = " ".join(str(x).lower() for x in brut.iloc[i].tolist())
                if "date" in ligne and "cip" in ligne:
                    t = brut.iloc[i + 1:].copy()
                    t.columns = [str(c).strip().lower() for c in brut.iloc[i]]
                    d = [c for c in t.columns if "date" in c][0]
                    c = [c for c in t.columns if c.startswith("cip")][0]
                    tables.append(t[[d, c]].set_axis(["date", "cip"], axis=1))
                    break
    s = pd.concat(tables)
    s["date"] = pd.to_datetime(s.date, errors="coerce")
    s["cip"] = s.cip.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s["cis"] = s.cip.map(cip.set_index("cip13").cis)
    lignes = len(s)
    s = s.dropna(subset=["cis", "date"])
    relies = len(s)
    s = s.drop_duplicates(["cis", "date"])
    return s, {"lignes": lignes, "relies": relies, "uniques": len(s)}


def taille_boite(texte):
    """'plaquette(s) de 30 comprime(s)' -> 30. Une boite de 90 ne vaut pas une boite de 30."""
    t = sans_accent(texte)
    unite = r"(comprime|gelule|sachet|capsule|ampoule|dose|lyophilisat|pastille|suppositoire|ovule|patch|dispositif|stylo|seringue|flacon|unidose|recipient|cartouche|film|gomme|implant|pilulier)"
    m = re.search(r"de\s+(\d+)\s+" + unite, t) or re.search(r"^(\d+)\s+" + unite, t)
    if not m:
        return 1
    n = int(m.group(1))
    m2 = re.match(r"^(\d+)\s+(plaquette|film|sachet|flacon|tube|boite|pilulier)", t)
    if m2 and int(m2.group(1)) > 1 and not t.startswith(m.group(0)):
        n *= int(m2.group(1))
    return n


def famille_forme(f):
    f = sans_accent(f)
    if re.search(r"comprime|gelule|capsule|lyophilisat|pastille", f):
        if re.search(r"liberation (prolongee|modifiee)", f):
            return "oral_LP"
        if re.search(r"orodispersible|effervescent|a sucer|lyophilisat", f):
            return "oral_special"
        return "oral"
    for motif, nom in [("buvable|sirop|gouttes", "buvable"), ("sachet|granule|poudre orale", "sachet"),
                       ("injectable|perfusion", "inj"), ("collyre|ophtalm", "oeil"), ("inhal|nebuli", "inhal"),
                       ("nasal", "nasal"), ("transdermique|patch", "patch"),
                       ("creme|pommade|gel|lotion|cutane|emulsion|mousse|shampooing", "peau"),
                       ("suppositoire|rectal|lavement", "rectal"), ("ovule|vaginal|intra-uterin", "vaginal")]:
        if re.search(motif, f):
            return nom
    return "autre_" + f.split(",")[0].split(" ")[0]


def dose_normalisee(s):
    s = sans_accent(s or "").replace(" ", "").replace(",", ".")
    m = re.match(r"^([\d.]+)(g|mg|microgrammes?|µg|ug|ui|u\.i\.|%)", s)
    if not m:
        return s
    v, u = float(m.group(1)), m.group(2)
    if u == "g":
        v, u = v * 1000, "mg"
    if u in ("microgrammes", "microgramme", "µg", "ug"):
        v, u = v / 1000, "mg"
    return f"{v:g}{'ui' if u == 'u.i.' else u}" + s[m.end():]


class Unions:
    """Regroupe les elements relies entre eux (composantes connexes)."""
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def construire_groupes(ref, compo, gener):
    """Meme traitement = meme substance + meme dose + meme forme, ou meme groupe generique,
    ou meme nom et dose (changement de code d'un meme produit)."""
    ref = ref.copy()
    ref["fam"] = ref.forme.map(famille_forme)
    c = compo[compo.cis.isin(ref.index)].copy()
    c["pri"] = (c.nature == "FT").astype(int)
    c = c.sort_values(["cis", "lien", "pri"], ascending=[True, True, False]).drop_duplicates(["cis", "lien"])
    c["dose"] = c.dosage.map(dose_normalisee)
    ref["dose"] = c.groupby("cis").dose.apply(lambda s: "+".join(sorted(s)))
    gen = gener[gener.type.isin(["0", "1", "4"])].drop_duplicates("cis").set_index("cis").grp
    u = Unions()
    for cis, r in ref.iterrows():
        u.union(f"I{cis}", f"K{r.atc}|{r.dose}|{r.fam}")
        u.union(f"I{cis}", "S" + re.sub(r"[^a-z0-9%]", "", sans_accent(r.nom).split(",")[0]) + "|" + r.fam)
        if cis in gen.index:
            u.union(f"I{cis}", f"G{gen[cis]}")
    racines = pd.Series({cis: u.find(f"I{cis}") for cis in ref.index})
    ref["traitement"] = "T" + racines.groupby(racines).ngroup().astype(str).str.zfill(5)
    return ref


def mesurer(A, volume, classes, calmes, seuil_gel=SEUIL_MANQUE, gel_max=12):
    """Boites attendues = niveau habituel x saison x effet du mois.  D = reel / attendu."""
    n, T = A.shape
    L, S, C = np.full((n, T), np.nan), np.ones((n, T)), np.full((n, T), np.nan)
    E, D = np.full((n, T), np.nan), np.full((n, T), np.nan)
    gel = np.zeros(n, int)
    temoins = calmes & (volume >= VOLUME_MIN)
    par_classe = {k: np.where(classes == k)[0] for k in pd.unique(classes)}
    for t in range(6, T):
        # 1. Niveau habituel : mediane des 12 derniers mois, gelee pendant un manque (12 mois maximum)
        niveau = np.nanmedian(A[:, max(0, t - 12):t], axis=1)
        if t > 6:
            bas = (D[:, t - 1] < seuil_gel) & (L[:, t - 1] > niveau) & (gel < gel_max)
            gel = np.where(bas, gel + 1, 0)
            niveau = np.where(bas, L[:, t - 1], niveau)
        L[:, t] = niveau
        # 2. Saison du medicament : meme periode l'an dernier / niveau de l'an dernier
        if t >= 13:
            an_dernier = np.nanmedian(A[:, max(0, t - 18):t - 6], axis=1)
            periode = A[:, t - 13:t - 10].mean(axis=1)
            S[:, t] = np.clip(np.where(an_dernier > 0, periode / an_dernier, 1.0), 0.25, 4)
        base = niveau * S[:, t]
        # 3. Effet du mois : variation mediane des medicaments sans alerte de la meme classe
        ratio = np.where(base > 0, A[:, t] / base, np.nan)
        coef = np.full(n, np.nanmedian(ratio[temoins]))
        for k, I in par_classe.items():
            J = I[temoins[I]]
            if len(J) >= 8:
                coef[I] = np.nanmedian(ratio[J])
        C[:, t] = coef
        E[:, t] = base * coef
        D[:, t] = np.where(E[:, t] > 0, A[:, t] / E[:, t], np.nan)
    return D, E, L, S, C


def lire_declarations(data):
    """Extrait data.ansm, facultatif : son nom peut etre 'Copie de declarations.csv'."""
    trouves = glob.glob(f"{data}/ansm_data/*declarations*.csv")
    return set(pd.read_csv(trouves[0], dtype=str).cis) if trouves else set()


def long(D, E, L, S, C, reel, ids, mois, niveau_nom, diviseur=None):
    """Passe une matrice medicaments x mois en table simple, une ligne par medicament et par mois."""
    d = np.ones(len(ids)) if diviseur is None else diviseur
    n, T = len(ids), len(mois)
    t = pd.DataFrame({
        "niveau": niveau_nom, "id": np.repeat(ids, T), "mois": np.tile(mois, n),
        "reel": np.round((reel / d[:, None]).ravel(), 0),
        "attendu": np.round((E / d[:, None]).ravel(), 0),
        "habituel": np.round((L / d[:, None]).ravel(), 0),
        "saison": np.round(S.ravel(), 3), "effet": np.round(C.ravel(), 3), "dispo": np.round(D.ravel(), 6)})
    return t.dropna(subset=["dispo"])


def calculer(data, sortie, inter):
    """De la donnee brute aux trois fichiers de resultats. Une a deux minutes."""
    os.makedirs(sortie, exist_ok=True)
    os.makedirs(inter, exist_ok=True)
    journal = []

    def noter(etape, action, pourquoi, avant, apres):
        journal.append({"Étape": etape, "Transformation": action, "Pourquoi": pourquoi, "Avant": avant, "Après": apres})

    cis = lire_bdpm(data, "CIS_bdpm.txt", "cis nom forme voie statut_amm proc etat date_amm statut_bdm eu titulaire surv".split())
    cip = lire_bdpm(data, "CIS_CIP_bdpm.txt", "cis cip7 pres statut etat date cip13 agrement taux p1 p2 p3 ind".split())
    compo = lire_bdpm(data, "CIS_COMPO_bdpm.txt", "cis forme code sub dosage ref nature lien x".split())
    gener = lire_bdpm(data, "CIS_GENER_bdpm.txt", "grp lib cis type tri".split())
    mitm = lire_bdpm(data, "CIS_MITM.txt", "cis atc nom tit exp date url".split())
    dispo = lire_bdpm(data, "CIS_CIP_Dispo_Spec.txt", "cis cip13 code statut debut maj remise url".split())
    pv, ref_m, st_m = lire_medicam(data)
    sig, st_s = lire_signalements(data, cip)
    mois = list(pv.columns)
    noter(1, "Empiler les fichiers Medic'AM", "Suivre chaque présentation dans le temps",
          f"{st_m['fichiers']} fichiers", f"{len(pv)} présentations sur {len(mois)} mois")
    noter(2, "Mettre à 0 les cases vides et les valeurs négatives", "Vide = rien vendu ; négatif = correction de remboursement",
          f"{st_m['vides']} vides, {st_m['negatifs']} négatives", "0 vide, 0 négative")
    pv.stack().rename("boites").reset_index().to_csv(f"{inter}/1_medicam_boites_par_mois.csv", index=False)
    sig[["date", "cip", "cis"]].sort_values("date").to_csv(f"{inter}/3_signalements_ansm_nettoyes.csv", index=False)

    # Boites -> unites, puis presentation -> specialite (CIS)
    c = cip[cip.cip13.isin(pv.index)].copy()
    c["taille"] = c.pres.map(taille_boite)
    b = pv.reindex(c.cip13)
    U = b.mul(c.taille.values, axis=0).groupby(c.cis.values).sum()
    B = b.groupby(c.cis.values).sum()

    noter(3, "Convertir les boîtes en unités et regrouper par spécialité (CIS)", "Une boîte de 90 ne vaut pas une boîte de 30",
          f"{len(c)} présentations reliées à la BDPM", f"{len(U)} spécialités")
    noter(4, "Nettoyer les signalements ANSM", "Sans code CIS on ne peut pas relier ; un doublon compterait deux fois",
          f"{st_s['lignes']} lignes", f"{st_s['uniques']} signalements uniques reliés")
    lg = pd.DataFrame({"cis": np.repeat(U.index.values, len(mois)), "mois": np.tile(mois, len(U)),
                       "boites": B.reindex(U.index).values.ravel(), "unites": U.values.ravel()})
    lg.to_csv(f"{inter}/2_unites_par_specialite.csv", index=False)

    # Referentiel
    c["total"] = pv.sum(axis=1).reindex(c.cip13).values
    gros = c.sort_values("total", ascending=False).drop_duplicates("cis").set_index("cis")
    ref = cis.set_index("cis")[["nom", "titulaire", "etat", "forme"]].reindex(U.index)
    ref["atc"] = ref_m.atc.reindex(gros.cip13.reindex(ref.index)).values
    ref = ref.dropna(subset=["nom"])
    U, B = U.loc[ref.index], B.loc[ref.index]
    ref["mitm"] = ref.index.isin(mitm.cis)
    ref["vaccin"] = ref.atc.astype(str).str.startswith("J07")
    ref = construire_groupes(ref, compo, gener)

    # Arrets de commercialisation : ce n'est pas une penurie
    c["arret"] = c.etat.str.contains("arr", case=False, na=False)
    vendu = c[c.total > 0].groupby("cis").agg(tous=("arret", "all"))
    ref["arrete"] = vendu.tous.reindex(ref.index).fillna(False).astype(bool) | ref.etat.ne("Commercialisée")
    ref["arret_traitement"] = ref.groupby("traitement").arrete.transform("all")

    noter(5, "Regrouper les spécialités en traitements", "Une autre marque peut servir le patient",
          f"{len(ref)} spécialités", f"{ref.traitement.nunique()} traitements")

    # Mesure aux deux niveaux
    touches = set(sig[sig.date.dt.year >= 2022].cis) | set(dispo.cis) | lire_declarations(data)
    Vb = np.nanmedian(B.values, axis=1)
    Db, Eb, Lb, Sb, Cb = mesurer(U.values.astype(float), Vb, ref.atc.str[:4].fillna("?").values, ~ref.index.isin(touches))
    UT, BT = U.groupby(ref.traitement).sum(), B.groupby(ref.traitement).sum()
    Vt = np.nanmedian(BT.values, axis=1)
    rt = ref.assign(v=Vb).sort_values("v", ascending=False).drop_duplicates("traitement").set_index("traitement").reindex(UT.index)
    calmes = ~ref.groupby("traitement").apply(lambda x: x.index.isin(touches).any()).reindex(UT.index).values
    Dt, Et, Lt, St, Ct = mesurer(UT.values.astype(float), Vt, rt.atc.str[:4].fillna("?").values, calmes)
    dernier_t, dernier_b = pd.Series(Dt[:, -1], index=UT.index), pd.Series(Db[:, -1], index=U.index)

    # Signes d'alerte (statuts ANSM ouverts = version en temps reel des declarations)
    part = pd.Series(Vb, index=ref.index)
    part = (part / part.groupby(ref.traitement).transform("sum")).fillna(0)
    statuts = dispo[dispo.statut.fillna("").str.startswith(("Rupture", "Tension"))]
    alerte = ref.index.isin(statuts.cis)
    s_alerte = (pd.Series(alerte * part.values, index=ref.index).groupby(ref.traitement).sum() >= 0.5).reindex(UT.index)
    x = pd.DataFrame({"t": ref.traitement, "l": ref.titulaire, "v": U.iloc[:, -12:].sum(axis=1)}).groupby(["t", "l"]).v.sum().reset_index()
    x["p"] = (x.v / x.groupby("t").v.transform("sum").replace(0, np.nan)) ** 2
    s_seul = (x.groupby("t").p.sum() >= 0.75).reindex(UT.index)
    d3 = pd.DataFrame(Dt[:, -3:], index=UT.index).clip(upper=1.2).mean(axis=1)
    en_champ = (Vt >= VOLUME_MIN) & ~rt.arret_traitement.values & ~rt.vaccin.values
    cl = pd.DataFrame({"k": rt.atc.str[:4].values, "bas": d3.values < SEUIL_MANQUE})[en_champ].groupby("k").bas.mean()
    s_classe = pd.Series(rt.atc.str[:4].map(cl).values, index=UT.index) >= 0.10
    s_non_mitm = ~ref.groupby("traitement").mitm.max().reindex(UT.index).astype(bool)
    points = 2 * s_alerte.astype(int) + 2 * s_classe.astype(int) + s_seul.astype(int) + s_non_mitm.astype(int)

    grp_arret = rt.arret_traitement.values
    noter(6, "Retirer les traitements arrêtés", "Un arrêt définitif n'est pas une pénurie", f"{len(UT)} traitements", f"{int((~grp_arret).sum())} traitements")
    noter(7, "Retirer les vaccins", "Ils suivent des campagnes", f"{int((~grp_arret).sum())} traitements", f"{int((~grp_arret & ~rt.vaccin.values).sum())} traitements")
    noter(8, f"Garder {VOLUME_MIN} boîtes par mois ou plus", "En dessous, le chiffre est trop instable",
          f"{int((~grp_arret & ~rt.vaccin.values).sum())} traitements", f"{int(en_champ.sum())} traitements")

    # Fichier 1 : les traitements
    T = pd.DataFrame({"traitement": UT.index, "medicament_principal": rt.nom.values, "atc": rt.atc.values,
                      "nb_specialites": ref.groupby("traitement").size().reindex(UT.index).values,
                      "boites_par_mois": Vt.round(0), "dispo": dernier_t.clip(upper=1).round(6).values,
                      "points": points.values, "signe_alerte": s_alerte.values, "signe_classe": s_classe.values,
                      "signe_seul": s_seul.values, "signe_non_mitm": s_non_mitm.values})[en_champ]
    T = T.dropna(subset=["dispo"])

    # Fichier 2 : tous les medicaments de la BDPM, y compris ceux qu'on ne peut pas calculer
    statut = dispo.sort_values("maj").drop_duplicates("cis", keep="last").set_index("cis").statut
    M = cis.set_index("cis")[["nom", "titulaire", "etat"]].copy()
    M["titulaire"] = M.titulaire.str.strip()
    M["statut_ansm"] = statut.reindex(M.index).fillna("")
    M["traitement"] = ref.traitement.reindex(M.index)
    M["boites_par_mois"] = pd.Series(Vb, index=ref.index).reindex(M.index).round(0)
    M["dispo_boite"] = dernier_b.clip(upper=1).round(6).reindex(M.index)
    boite_ok = pd.Series((Vb >= VOLUME_MIN) & ~ref.arrete.values & ~ref.vaccin.values, index=ref.index).reindex(M.index).fillna(False).astype(bool)
    M["boite_calculee"] = boite_ok & M.dispo_boite.notna()
    M["traitement_calcule"] = M.traitement.isin(T.traitement)
    raison = pd.Series("", index=M.index)
    raison[M.boites_par_mois.isna()] = "Absent des remboursements en ville : médicament d'hôpital, non remboursé ou plus vendu."
    raison[M.boites_par_mois.notna() & ~M.traitement_calcule] = f"Moins de {VOLUME_MIN:,} boîtes par mois : le chiffre serait trop instable.".replace(",", " ")
    raison[ref.index[ref.arret_traitement]] = "Commercialisation arrêtée : ce n'est pas une rupture temporaire."
    raison[ref.index[ref.vaccin]] = "Vaccin : ses ventes suivent des campagnes, pas la disponibilité."
    M["raison_hors_champ"] = raison.where(~M.traitement_calcule, "")

    # Fichier 3 : les series mensuelles, en boites pour la boite, en unites pour le traitement
    garder_b = M.boite_calculee.reindex(ref.index).values
    taille = (U.sum(axis=1) / B.sum(axis=1).replace(0, np.nan)).fillna(1).values
    s1 = long(Db[garder_b], Eb[garder_b], Lb[garder_b], Sb[garder_b], Cb[garder_b], U.values[garder_b],
              U.index[garder_b].values, mois, "boite", taille[garder_b])
    garder_t = UT.index.isin(T.traitement)
    s2 = long(Dt[garder_t], Et[garder_t], Lt[garder_t], St[garder_t], Ct[garder_t], UT.values[garder_t],
              UT.index[garder_t].values, mois, "traitement")

    # Tableaux intermediaires et resultat final lisibles dans Excel
    ref.reset_index(names="cis")[["cis", "nom", "titulaire", "atc", "fam", "dose", "traitement", "mitm", "vaccin", "arrete"]].to_csv(
        f"{inter}/4_medicaments_et_traitements.csv", index=False)
    s2.drop(columns="niveau").rename(columns={"id": "traitement", "reel": "delivre", "habituel": "niveau_habituel", "effet": "effet_du_mois", "dispo": "disponibilite"}).to_csv(
        f"{inter}/5_traitements_par_mois.csv", index=False)
    prevu = 1 - (1 - T.dispo) * FACTEUR_MENSUEL ** 3
    F = pd.DataFrame({"traitement": T.traitement, "medicament_principal": T.medicament_principal, "atc": T.atc,
                      "nb_specialites": T.nb_specialites, "boites_par_mois": T.boites_par_mois,
                      "dispo_observee_%": (T.dispo * 100).round(1), "dispo_prevue_3mois_%": (prevu * 100).round(1),
                      "niveau": np.where(prevu >= SEUIL_TENDU, "Disponible", np.where(prevu >= SEUIL_MANQUE, "Tendu", "Manque")),
                      "points_alerte": T.points})
    F["risque_chute_6mois"] = np.where(F.niveau != "Disponible", "", np.where(F.points_alerte >= 4, "Eleve", np.where(F.points_alerte >= 2, "Moyen", "Faible")))
    F.sort_values("dispo_prevue_3mois_%").to_csv(f"{sortie}/disponibilite_traitements.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    Mb = M[M.boite_calculee].reset_index()
    pb = 1 - (1 - Mb.dispo_boite) * FACTEUR_MENSUEL ** 3
    G = pd.DataFrame({"cis": Mb.cis, "medicament": Mb.nom, "laboratoire": Mb.titulaire, "traitement": Mb.traitement,
                      "boites_par_mois": Mb.boites_par_mois, "dispo_boite_observee_%": (Mb.dispo_boite * 100).round(1),
                      "dispo_boite_prevue_3mois_%": (pb * 100).round(1), "statut_ansm": Mb.statut_ansm})
    G["dispo_traitement_prevue_3mois_%"] = G.traitement.map(F.set_index("traitement")["dispo_prevue_3mois_%"])
    G.sort_values("dispo_boite_prevue_3mois_%").to_csv(f"{sortie}/disponibilite_boites.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    noter(9, "Appliquer la formule", "Disponibilité = délivré ÷ attendu ; prévision = 100 % − manque × 0,9^h",
          f"{int(en_champ.sum())} traitements", f"{len(T)} traitements et {len(G)} boîtes calculés")
    pd.DataFrame(journal).to_csv(f"{inter}/journal_pipeline.csv", index=False, encoding="utf-8-sig")

    T.to_csv(f"{sortie}/traitements.csv.gz", index=False)
    M.reset_index().to_csv(f"{sortie}/medicaments.csv.gz", index=False)
    pd.concat([s1, s2]).to_csv(f"{sortie}/series.csv.gz", index=False)
    meta = {"dernier_mois": mois[-1], "premier_mois": mois[0], "nb_mois": len(mois), "nb_traitements": int(len(T)),
            "nb_boites": int(M.boite_calculee.sum()), "calcule_le": time.strftime("%d/%m/%Y %H:%M")}
    json.dump(meta, open(f"{sortie}/infos.json", "w"), ensure_ascii=False)
    return meta


if __name__ == "__main__":
    ici = os.path.dirname(os.path.abspath(__file__))
    print(calculer(f"{ici}/donnees", f"{ici}/resultats", f"{ici}/donnees_transformees"))
