# =============================================================================
# AvailScore : mise à jour automatique des données
#
# donnees/ (fichiers bruts)  ->  exports_power_bi/ (7 tables)  ->  resultats/ (fichiers propres)
#
# 1. (option) télécharger la dernière BDPM sur base-donnees-publique.medicaments.gouv.fr
# 2. relire les fichiers bruts BDPM, ANSM et Medic'AM, vérifier leur format,
#    et réécrire les 7 tables au format des exports Power BI
# 3. relancer python/1_nettoyage_final.py
# 4. noter l'état des fichiers (manifeste.json) : au prochain lancement, l'application
#    voit toute seule si un fichier de donnees/ a été ajouté, remplacé ou supprimé.
# =============================================================================

import glob
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.request
from datetime import datetime
from io import StringIO

import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
DONNEES = os.path.join(ICI, "donnees")
EXPORTS = os.path.join(ICI, "exports_power_bi")
RESULTATS = os.path.join(ICI, "resultats")
CACHE = os.path.join(ICI, "cache_import")
MANIFESTE = os.path.join(RESULTATS, "manifeste.json")

URL_BDPM = "https://base-donnees-publique.medicaments.gouv.fr/download/file/{}"
URL_MEDICAM = "https://www.data.gouv.fr/api/1/datasets/536999d3a3a729239d20533a/"
URL_ANSM = ("https://ansm.sante.fr/page/"
            "medicaments-ayant-fait-lobjet-dun-signalement-de-rupture-ou-de-risque-de-rupture-de-stock")

# Fichier BDPM -> (nombre minimal de colonnes, position du code CIS, colonnes gardées {position: nom})
BDPM = {
    "CIS_bdpm.txt": (11, 0, {0: "cis", 1: "nom", 2: "forme", 3: "voie", 6: "etat", 10: "titulaire"}),
    "CIS_CIP_bdpm.txt": (7, 0, {0: "cis", 1: "cip7", 2: "presentation", 4: "etat_commercialisation", 6: "cip13"}),
    "CIS_COMPO_bdpm.txt": (8, 0, {0: "cis", 3: "substance", 4: "dosage", 6: "nature", 7: "lien"}),
    "CIS_GENER_bdpm.txt": (4, 2, {0: "groupe", 2: "cis", 3: "type"}),
    "CIS_MITM.txt": (2, 0, {0: "cis", 1: "atc"}),
    "CIS_CIP_Dispo_Spec.txt": (7, 0, {0: "cis", 1: "cip13", 2: "code_statut", 3: "statut", 4: "date_debut",
                                      5: "date_maj", 6: "date_remise", 7: "lien"}),
}


class DonneesNonConformes(Exception):
    """Un fichier n'a pas le format attendu : on s'arrête avant d'écraser quoi que ce soit."""


def sans_accent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


# ------------------------------------------------------------------ BDPM
def lire_bdpm(chemin, nom=None):
    """Lit un fichier BDPM (tabulations, sans titre, UTF-8 ou Windows-1252 selon les fichiers)
    et vérifie qu'il a toujours la même forme."""
    nom = nom or os.path.basename(chemin)
    n_min, col_cis, garder = BDPM[nom]
    brut = open(chemin, "rb").read()
    try:
        texte = brut.decode("utf-8")
    except UnicodeDecodeError:
        texte = brut.decode("cp1252", errors="replace")
    t = pd.read_csv(StringIO(texte), sep="\t", header=None, dtype=str, quoting=3,
                    keep_default_na=False, na_values=[""])
    if len(t) == 0:
        raise DonneesNonConformes(f"{nom} est vide.")
    if t.shape[1] < n_min:
        raise DonneesNonConformes(f"{nom} : {t.shape[1]} colonnes au lieu d'au moins {n_min}. "
                                  "Le format de la BDPM a changé.")
    ok = t[col_cis].fillna("").str.strip().str.fullmatch(r"\d{8}").mean()
    if ok < 0.95:
        raise DonneesNonConformes(f"{nom} : la colonne {col_cis + 1} ne contient pas des codes CIS "
                                  f"({ok:.0%} seulement ressemblent à un CIS).")
    t = t[[c for c in garder if c < t.shape[1]]].rename(columns=garder)
    return t.apply(lambda c: c.str.strip())


def telecharger_bdpm(avance=lambda f, m: None):
    """Télécharge les 6 fichiers BDPM. Chaque fichier est vérifié AVANT de remplacer l'ancien."""
    dossier = os.path.join(DONNEES, "bdpm")
    os.makedirs(dossier, exist_ok=True)
    changes = []
    for i, nom in enumerate(BDPM):
        avance(i / len(BDPM), f"Téléchargement BDPM : {nom}")
        req = urllib.request.Request(URL_BDPM.format(nom), headers={"User-Agent": "AvailScore"})
        with urllib.request.urlopen(req, timeout=120) as r:
            contenu = r.read()
        if contenu[:15].lstrip().lower().startswith((b"<!doctype", b"<html")):
            raise DonneesNonConformes(f"{nom} : le site a renvoyé une page web au lieu du fichier.")
        temp = os.path.join(dossier, nom + ".nouveau")
        with open(temp, "wb") as f:
            f.write(contenu)
        try:
            lire_bdpm(temp, nom)
        except Exception:
            os.remove(temp)
            raise
        final = os.path.join(dossier, nom)
        ancien = open(final, "rb").read() if os.path.exists(final) else None
        if ancien != contenu:
            os.replace(temp, final)
            changes.append(nom)
        else:
            os.remove(temp)
    avance(1, "BDPM téléchargée")
    return changes


def telecharger(url):
    req = urllib.request.Request(url, headers={"User-Agent": "AvailScore"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def telecharger_medicam(depuis=2021, avance=lambda f, m: None):
    """Ajoute les semestres Medic'AM (série mensuelle) publiés sur data.gouv.fr qui manquent dans
    donnees/medicam/. Seuls les fichiers Excel de l'archive sont extraits, sous un nom choisi ici."""
    import zipfile
    from io import BytesIO
    dossier = os.path.join(DONNEES, "medicam")
    os.makedirs(dossier, exist_ok=True)
    api = json.loads(telecharger(URL_MEDICAM))
    presents = {os.path.basename(f)[:12] for f in glob.glob(os.path.join(dossier, "*"))}
    a_prendre = {}
    for r in api.get("resources", []):
        m = re.search(r"/((\d{4})-\d{2}-a-\d{2})_medic-am[^/]*serie-mensuelle", r.get("url", ""))
        if m and int(m.group(2)) >= depuis and m.group(1) not in presents:
            a_prendre[m.group(1)] = r["url"]
    ajoutes = []
    for i, (periode, url) in enumerate(sorted(a_prendre.items())):
        avance(i / max(1, len(a_prendre)), f"Téléchargement Medic'AM {periode}")
        contenu = telecharger(url)
        if url.lower().endswith(".zip"):
            with zipfile.ZipFile(BytesIO(contenu)) as z:
                excel = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))]
                if not excel:
                    raise DonneesNonConformes(f"Medic'AM {periode} : pas de fichier Excel dans l'archive.")
                ext = os.path.splitext(excel[0])[1].lower()
                contenu = z.read(excel[0])
        else:
            ext = os.path.splitext(url)[1].lower()
        final = os.path.join(dossier, f"{periode}_medic-am-par-type-de-prescripteur_serie-mensuelle{ext}")
        with open(final, "wb") as f:
            f.write(contenu)
        try:
            lire_medicam_fichier(final)                  # vérifie le format
        except Exception:
            os.remove(final)
            raise
        ajoutes.append(os.path.basename(final))
    avance(1, "Medic'AM à jour")
    return ajoutes


def telecharger_ansm(avance=lambda f, m: None):
    """Ajoute les listes ANSM des ruptures et risques de rupture publiées sur la page de l'ANSM
    qui manquent dans donnees/ansm/ (une liste par an, publiée au printemps de l'année suivante)."""
    dossier = os.path.join(DONNEES, "ansm")
    os.makedirs(dossier, exist_ok=True)
    page = telecharger(URL_ANSM).decode("utf-8", errors="replace")
    liens = sorted(set(re.findall(r'href="([^"]+?\.xlsx?)"', page)))
    liens = [l if l.startswith("http") else "https://ansm.sante.fr" + l for l in liens
             if re.search(r"rupture|rs-liste", l, re.I)]
    presents = {os.path.basename(f) for f in glob.glob(os.path.join(dossier, "*"))}
    ajoutes = []
    for i, url in enumerate(liens):
        nom = os.path.basename(url)
        if nom in presents:
            continue
        avance(i / max(1, len(liens)), f"Téléchargement ANSM : {nom}")
        with open(os.path.join(dossier, nom), "wb") as f:
            f.write(telecharger(url))
        ajoutes.append(nom)
    avance(1, "Listes ANSM à jour")
    return ajoutes


def rafraichir_statuts(age_max_heures=12):
    """Le statut actuel (rupture, tension, remise à disposition) change tous les jours : on recharge
    seulement ce petit fichier BDPM s'il a plus de age_max_heures, sans refaire tout le traitement.
    Renvoie la date du fichier utilisé. En cas d'échec (pas d'internet), on garde l'ancien."""
    nom = "CIS_CIP_Dispo_Spec.txt"
    chemin = os.path.join(DONNEES, "bdpm", nom)
    sortie = os.path.join(RESULTATS, "disponibilite_ansm.csv")
    age = (datetime.now().timestamp() - os.path.getmtime(chemin)) / 3600 if os.path.exists(chemin) else 1e9
    if age > age_max_heures:
        try:
            contenu = telecharger(URL_BDPM.format(nom))
            temp = chemin + ".nouveau"
            with open(temp, "wb") as f:
                f.write(contenu)
            lire_bdpm(temp, nom)
            os.replace(temp, chemin)
            m = lire_manifeste()
            if m is not None:                            # pas besoin de refaire tout le traitement
                m["fichiers"][os.path.relpath(chemin, ICI).replace("\\", "/")] = empreinte()[
                    os.path.relpath(chemin, ICI).replace("\\", "/")]
                with open(MANIFESTE, "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, indent=1)
        except Exception:
            if os.path.exists(chemin + ".nouveau"):
                os.remove(chemin + ".nouveau")
    if os.path.exists(chemin) and (not os.path.exists(sortie) or os.path.getmtime(sortie) < os.path.getmtime(chemin)):
        lire_bdpm(chemin).to_csv(sortie, index=False)
    return datetime.fromtimestamp(os.path.getmtime(chemin)) if os.path.exists(chemin) else None


# ------------------------------------------------------------------ ANSM
def lire_historique_ansm(dossier):
    """Fichier Data.ansm « historique ruptures » : pour chaque déclaration, le TYPE
    (rupture ou risque de rupture) et la cause. Sert à séparer les deux dans la prédiction."""
    morceaux = []
    for f in sorted(glob.glob(os.path.join(dossier, "*.xls*"))):
        t = pd.read_excel(f, dtype={"cis": str})
        t.columns = [sans_accent(c).strip() for c in t.columns]
        if not {"cis", "classification", "date"} <= set(t.columns):
            raise DonneesNonConformes(f"{os.path.basename(f)} : colonnes cis / classification / date attendues.")
        morceaux.append(t[["cis", "date", "classification"] + (["cause"] if "cause" in t.columns else [])])
    if not morceaux:
        return pd.DataFrame(columns=["cis", "date", "classification", "cause"])
    h = pd.concat(morceaux, ignore_index=True)
    h["date"] = pd.to_datetime(h.date, errors="coerce").dt.strftime("%Y-%m-%d")
    h["classification"] = h.classification.str.lower().str.strip()
    return h.dropna(subset=["cis", "date"]).drop_duplicates()


def nom_court(s):
    """Nom de spécialité comparable entre l'ANSM et la BDPM : minuscules, sans accents ni ponctuation."""
    return re.sub(r"[^a-z0-9]+", " ", sans_accent(s)).strip()


def lire_ansm(fichiers, cip_vers_cis, cip7_vers_cip13, nom_vers_cis=None):
    """Les listes ANSM (une feuille par année ou une feuille par fichier).
    On repère la ligne de titres, la colonne de date et la colonne CIP, quel que soit leur ordre.
    Une feuille sans code CIP (2021) est reliée aux médicaments par le nom exact de la spécialité.
    Renvoie aussi le nombre de lignes reliées par le nom, et les feuilles ignorées."""
    est_cip = lambda x: re.fullmatch(r"(code )?cip ?(13)?", x.strip())
    est_nom = lambda x: x.strip() in ("specialite", "nom", "nom de la specialite", "denomination")
    morceaux, ignores, par_nom = [], [], 0
    for f in fichiers:
        for feuille, t in pd.read_excel(f, sheet_name=None, header=None, dtype=object).items():
            titres = None
            for i in range(min(15, len(t))):
                ligne = [sans_accent(x) if pd.notna(x) else "" for x in t.iloc[i]]
                if any("date" in x for x in ligne) and any(est_cip(x) or est_nom(x) for x in ligne):
                    titres = i
                    break
            if titres is None:
                ignores.append(f"{os.path.basename(f)} / {feuille}")
                continue
            ligne = [sans_accent(x).strip() if pd.notna(x) else "" for x in t.iloc[titres]]
            c_date = next(i for i, x in enumerate(ligne) if "date" in x)
            c_cip = next((i for i, x in enumerate(ligne) if est_cip(x)), None)
            if c_cip is not None:
                d = t.iloc[titres + 1:, [c_date, c_cip]]
                d.columns = ["date", "cip13"]
                d["cis_nom"] = None
            elif nom_vers_cis is not None:
                c_nom = next(i for i, x in enumerate(ligne) if est_nom(x))
                d = t.iloc[titres + 1:, [c_date, c_nom]]
                d.columns = ["date", "nom"]
                d["cis_nom"] = d.nom.map(lambda x: nom_vers_cis.get(nom_court(x)) if pd.notna(x) else None)
                d["cip13"] = "-"
                par_nom += int(d.cis_nom.notna().sum())
                ignores.append(f"{os.path.basename(f)} / {feuille} : sans CIP, {d.cis_nom.notna().mean():.0%} "
                               "des lignes reliées par le nom exact")
                d = d[["date", "cip13", "cis_nom"]]
            else:
                ignores.append(f"{os.path.basename(f)} / {feuille}")
                continue
            morceaux.append(d)
    if not morceaux:
        raise DonneesNonConformes("Aucune liste ANSM lisible dans donnees/ansm/ (il faut une colonne date et une colonne CIP).")
    s = pd.concat(morceaux, ignore_index=True).dropna(subset=["date", "cip13"])
    # Date : vraie date Excel, ou texte '04/01/2022' (jour d'abord), ou '2022-01-04'
    texte = s.date.astype(str).str.strip()
    iso = pd.to_datetime(texte.str[:10], format="%Y-%m-%d", errors="coerce")
    fr = pd.to_datetime(texte.str[:10], format="%d/%m/%Y", errors="coerce")
    s["date"] = iso.fillna(fr)
    s = s.dropna(subset=["date"])
    cip = s.cip13.map(lambda x: re.sub(r"\D", "", str(int(x)) if isinstance(x, float) else str(x)))
    s["cip13"] = cip.where(cip.str.len() != 7, cip.map(cip7_vers_cip13))
    s["cis"] = s.cip13.map(cip_vers_cis).fillna(s.cis_nom)
    s = s[(s.cip13.str.len() == 13) | s.cis_nom.notna()]
    s.loc[s.cip13.str.len() != 13, "cip13"] = None
    s["date"] = s.date.dt.strftime("%Y-%m-%d")
    return s.sort_values("date")[["date", "cip13", "cis"]], ignores


# ------------------------------------------------------------------ Medic'AM
def lire_medicam_fichier(chemin):
    """Un fichier Medic'AM : la feuille 'tous prescripteurs', une colonne 'Nombre de boites
    remboursées AAAA-MM' par mois. Gardé en cache tant que le fichier ne change pas."""
    os.makedirs(CACHE, exist_ok=True)
    st = os.stat(chemin)
    cle = f"{os.path.basename(chemin)}.{st.st_size}.{int(st.st_mtime)}.csv"
    cache = os.path.join(CACHE, cle)
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype=str)

    feuilles = pd.ExcelFile(chemin).sheet_names
    feuille = next((f for f in feuilles if "tous" in sans_accent(f)), None) \
        or next((f for f in feuilles if "lisez" not in sans_accent(f)), None)
    t = pd.read_excel(chemin, sheet_name=feuille, dtype=object)
    titres = {c: sans_accent(c).replace("\n", " ") for c in t.columns}
    c_cip = next((c for c, x in titres.items() if x.strip() == "cip13"), None)
    c_nom = next((c for c, x in titres.items() if x.strip() == "nom court"), None)
    c_atc = next((c for c, x in titres.items() if re.fullmatch(r"code\s+atc", x.strip())), None)
    mois = {c: re.search(r"(\d{4}-\d{2})", x).group(1) for c, x in titres.items()
            if "nombre de boites" in x and re.search(r"\d{4}-\d{2}", x)}
    if not (c_cip and c_nom and c_atc and mois):
        raise DonneesNonConformes(f"{os.path.basename(chemin)} : colonnes CIP13 / NOM COURT / Code ATC / "
                                  "« Nombre de boites remboursées AAAA-MM » introuvables. Le format Medic'AM a changé.")
    t = t[[c_cip, c_nom, c_atc] + list(mois)].rename(columns={c_cip: "cip13", c_nom: "nom_court", c_atc: "atc", **mois})
    t = t.dropna(subset=["cip13"])
    t["cip13"] = pd.to_numeric(t.cip13, errors="coerce").astype("Int64").astype(str)
    t = t[t.cip13.str.len() == 13]
    m = t.melt(id_vars=["cip13", "nom_court", "atc"], var_name="mois", value_name="boites")
    b = pd.to_numeric(m.boites, errors="coerce").fillna(0).clip(lower=0)
    m["boites"] = b.map(lambda x: str(int(x)) if x == int(x) else str(x))
    for vieux in glob.glob(os.path.join(CACHE, os.path.basename(chemin) + ".*.csv")):
        os.remove(vieux)
    m.to_csv(cache, index=False)
    return m


# ------------------------------------------------------------------ état des fichiers
def empreinte():
    """Taille et date de chaque fichier brut."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DONNEES, "*", "*"))):
        if os.path.isfile(f) and not f.endswith(".nouveau"):
            s = os.stat(f)
            out[os.path.relpath(f, ICI).replace("\\", "/")] = [s.st_size, int(s.st_mtime)]
    return out


def lire_manifeste():
    try:
        with open(MANIFESTE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def changements():
    """Ce qui a changé dans donnees/ depuis le dernier traitement : {'ajoutés': [...], 'modifiés': [...], 'supprimés': [...]}.
    None si les données n'ont jamais été traitées par ce module."""
    m = lire_manifeste()
    if m is None:
        return None
    avant, maintenant = m.get("fichiers", {}), empreinte()
    return {"ajoutés": sorted(set(maintenant) - set(avant)),
            "modifiés": sorted(f for f in set(maintenant) & set(avant) if maintenant[f] != avant[f]),
            "supprimés": sorted(set(avant) - set(maintenant))}


def resume():
    """Quelques chiffres pour comparer avant / après une mise à jour."""
    r = {}
    try:
        f = pd.read_csv(os.path.join(RESULTATS, "fiches_medicaments.csv"), dtype={"cis": str})
        v = pd.read_csv(os.path.join(RESULTATS, "ventes_par_medicament.csv"), dtype={"cis": str}, usecols=["cis", "mois"])
        s = pd.read_csv(os.path.join(RESULTATS, "signalements_propres.csv"), dtype=str)
        r["Médicaments (CIS) avec des ventes"] = len(f)
        r["dont arrêtés"] = int(f.arrete.sum())
        r["dont MITM"] = int(f.mitm.sum())
        r["Mois de ventes Medic'AM"] = f"{v.mois.min()} à {v.mois.max()}"
        r["Signalements ANSM reliés"] = len(s)
        r["Dernier signalement ANSM"] = s.date.max()
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    try:
        d = pd.read_csv(os.path.join(RESULTATS, "disponibilite_ansm.csv"), dtype=str)
        for statut, n in d.drop_duplicates("cis").statut.value_counts().items():
            r[f"Statut ANSM actuel : {statut}"] = int(n)
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    return r


# ------------------------------------------------------------------ le traitement complet
def tout_mettre_a_jour(avance=lambda f, m: None, telecharger=False):
    """Refait toute la chaîne. avance(fraction 0..1, message) sert à la barre de progression.
    Renvoie un rapport : fichiers changés, chiffres avant/après, journal du nettoyage."""
    rapport = {"avant": resume(), "changements": changements(), "telecharges": None, "avertissements": []}

    if telecharger:
        rapport["telecharges"] = (telecharger_bdpm(lambda f, m: avance(0.05 * f, m))
                                  + telecharger_ansm(lambda f, m: avance(0.05 + 0.03 * f, m))
                                  + telecharger_medicam(avance=lambda f, m: avance(0.08 + 0.07 * f, m)))

    # BDPM -> Specialites, Presentations, Compositions, Generiques, MITM (+ disponibilités)
    tables = {}
    for i, nom in enumerate(BDPM):
        avance(0.15 + 0.15 * i / len(BDPM), f"Lecture et vérification de la BDPM : {nom}")
        chemin = os.path.join(DONNEES, "bdpm", nom)
        if not os.path.exists(chemin):
            raise DonneesNonConformes(f"Il manque donnees/bdpm/{nom}.")
        tables[nom] = lire_bdpm(chemin)
    pres = tables["CIS_CIP_bdpm.txt"]
    cip_vers_cis = pres.dropna(subset=["cip13"]).drop_duplicates("cip13").set_index("cip13").cis
    cip7_vers_cip13 = pres.dropna(subset=["cip7", "cip13"]).drop_duplicates("cip7").set_index("cip7").cip13
    spe = tables["CIS_bdpm.txt"].assign(cle=lambda x: x.nom.map(nom_court))
    nom_vers_cis = spe.drop_duplicates("cle", keep=False).set_index("cle").cis.to_dict()   # noms sans ambiguïté

    # ANSM -> Signalements
    avance(0.32, "Lecture des listes ANSM (ruptures et risques de rupture)")
    fichiers_ansm = sorted(glob.glob(os.path.join(DONNEES, "ansm", "*.xls*")))
    signal, ignores = lire_ansm(fichiers_ansm, cip_vers_cis, cip7_vers_cip13, nom_vers_cis)
    if ignores:
        rapport["avertissements"].append("Listes ANSM : " + " ; ".join(ignores))
    historique = lire_historique_ansm(os.path.join(DONNEES, "ansm_historique"))

    # Medic'AM -> Medicam
    fichiers_medicam = sorted(glob.glob(os.path.join(DONNEES, "medicam", "*.xls*")))
    if not fichiers_medicam:
        raise DonneesNonConformes("Aucun fichier Medic'AM dans donnees/medicam/.")
    morceaux = []
    for i, f in enumerate(fichiers_medicam):
        avance(0.40 + 0.35 * i / len(fichiers_medicam), f"Lecture Medic'AM : {os.path.basename(f)}")
        morceaux.append(lire_medicam_fichier(f))
    medicam = pd.concat(morceaux, ignore_index=True)
    # Deux fichiers pour le même mois : le plus récent (dernier dans l'ordre des noms) l'emporte.
    medicam = medicam.drop_duplicates(["cip13", "mois"], keep="last").sort_values(["mois", "cip13"])
    medicam["cis"] = medicam.cip13.map(cip_vers_cis)

    # Écrire les 7 tables, au même format que les exports Power BI
    avance(0.76, "Écriture des 7 tables (format des exports Power BI)")
    os.makedirs(EXPORTS, exist_ok=True)
    os.makedirs(RESULTATS, exist_ok=True)
    sorties = {
        "Medicam": medicam[["cip13", "nom_court", "atc", "mois", "boites", "cis"]],
        "Signalements": signal,
        "Presentations": pres[["cis", "cip13", "presentation", "etat_commercialisation"]],
        "Specialites": tables["CIS_bdpm.txt"],
        "Compositions": tables["CIS_COMPO_bdpm.txt"],
        "Generiques": tables["CIS_GENER_bdpm.txt"][["groupe", "cis", "type"]],
        "MITM": tables["CIS_MITM.txt"],
    }
    for nom, t in sorties.items():
        t.to_csv(os.path.join(EXPORTS, nom + ".csv"), index=False)
    tables["CIS_CIP_Dispo_Spec.txt"].to_csv(os.path.join(RESULTATS, "disponibilite_ansm.csv"), index=False)
    historique.to_csv(os.path.join(RESULTATS, "historique_types_ansm.csv"), index=False)

    # Le nettoyage habituel
    avance(0.80, "Nettoyage (1_nettoyage_final.py) : unités, traitements, arrêts, vaccins")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    res = subprocess.run([sys.executable, "1_nettoyage_final.py"], cwd=os.path.join(ICI, "python"),
                         capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    rapport["journal"] = res.stdout
    if res.returncode != 0:
        raise RuntimeError("Le nettoyage a échoué :\n" + res.stderr[-3000:])

    avance(0.97, "Enregistrement de l'état des données")
    with open(MANIFESTE, "w", encoding="utf-8") as f:
        json.dump({"date_traitement": datetime.now().isoformat(timespec="seconds"), "fichiers": empreinte()},
                  f, ensure_ascii=False, indent=1)
    rapport["apres"] = resume()
    avance(1, "Données à jour")
    return rapport


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Met à jour les données d'AvailScore.")
    p.add_argument("--telecharger", action="store_true",
                   help="télécharger d'abord la dernière BDPM, les listes ANSM et les semestres Medic'AM manquants")
    a = p.parse_args()
    r = tout_mettre_a_jour(lambda f, m: print(f"[{f:4.0%}] {m}"), telecharger=a.telecharger)
    print(r["journal"])
    for k in r["apres"]:
        print(f"{k:40s} {r['avant'].get(k, '-')!s:>25} -> {r['apres'][k]}")
