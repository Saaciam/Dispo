# =============================================================================
# ÉTAPE PYTHON 1 : de la version nettoyée par Power BI à la version complète
#
# Entrée  : les 7 fichiers exportés de Power BI, dans exports_power_bi/
#           Medicam, Signalements, Presentations, Specialites, Compositions, Generiques, MITM
# Sortie  : 3 fichiers propres, dans resultats/
#           ventes_par_medicament.csv    une ligne par médicament (CIS) et par mois
#           signalements_propres.csv     une ligne par signalement relié à un médicament
#           fiches_medicaments.csv       une ligne par médicament : nom, ATC, MITM, traitement...
#
# Ce que Power BI a déjà fait : lire, empiler, dépivoter, vides et négatifs à 0,
# codes en texte, doublons, fusion CIP13 -> CIS.
# Ce qu'on fait ici (plus simple en Python) : taille des boîtes, regroupement par
# médicament, regroupement en traitements, arrêts et vaccins.
#
# Lancer :  python 1_nettoyage_final.py      (depuis le dossier python/)
# =============================================================================

import os
import re
import unicodedata
import pandas as pd

ENTREE = "../exports_power_bi"
SORTIE = "../resultats"
os.makedirs(SORTIE, exist_ok=True)


# %% ÉTAPE 1 : lire les exports de Power BI
def lire(nom):
    """Lit un export, quel que soit l'outil qui l'a écrit (DAX Studio, Excel...) :
    séparateur , ou ; , titres du genre Medicam[cip13], tout en texte."""
    chemin = f"{ENTREE}/{nom}.csv"
    with open(chemin, encoding="utf-8-sig") as f:
        sep = ";" if f.readline().count(";") > 0 else ","
    t = pd.read_csv(chemin, sep=sep, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    t.columns = [re.sub(r"^.*\[|\]$", "", c).strip() for c in t.columns]
    return t

def nombre(colonne):
    """'1 234,0' ou '1234.0' -> 1234.0"""
    return pd.to_numeric(colonne.str.replace(" ", "").str.replace(",", "."), errors="coerce").fillna(0)

medicam = lire("Medicam")
signal = lire("Signalements")
pres = lire("Presentations")
spe = lire("Specialites")
compo = lire("Compositions")
gener = lire("Generiques")
mitm = lire("MITM")
medicam["boites"] = nombre(medicam.boites)
print("Lu :", len(medicam), "lignes Medic'AM,", len(signal), "signalements")


# %% ÉTAPE 2 : retirer les boîtes qu'on ne sait pas relier à un médicament
# Ce sont surtout de vieux codes qui ne sont plus dans la BDPM.
# Exemple : un CIP13 retiré en 2019 mais encore remboursé quelques fois.
sans_cis = medicam.cis.isna()
print(f"Boîtes sans CIS : {sans_cis.mean():.1%} des lignes, "
      f"{medicam.boites[sans_cis].sum() / medicam.boites.sum():.2%} des boîtes -> retirées")
medicam = medicam[~sans_cis]


# %% ÉTAPE 3 : convertir les boîtes en unités
# Une boîte de 90 comprimés ne vaut pas une boîte de 30.
# Exemple : "plaquette(s) de 30 comprimé(s)" -> 30 ; "3 plaquette(s) de 28 comprimé(s)" -> 84.
def sans_accent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()

def taille_boite(texte):
    t = sans_accent(texte)
    unite = (r"(comprime|gelule|sachet|capsule|ampoule|dose|lyophilisat|pastille|suppositoire|ovule|patch|"
             r"dispositif|stylo|seringue|flacon|unidose|recipient|cartouche|film|gomme|implant|pilulier)")
    m = re.search(r"de\s+(\d+)\s+" + unite, t) or re.search(r"^(\d+)\s+" + unite, t)
    if not m:
        return 1                                   # on ne sait pas lire : 1 boîte = 1 unité
    n = int(m.group(1))
    m2 = re.match(r"^(\d+)\s+(plaquette|film|sachet|flacon|tube|boite|pilulier)", t)
    if m2 and int(m2.group(1)) > 1 and not t.startswith(m.group(0)):
        n *= int(m2.group(1))                      # "3 plaquettes de 28" = 84
    return n

taille = pres.set_index("cip13").presentation.map(taille_boite)
medicam["unites"] = medicam.boites * medicam.cip13.map(taille).fillna(1)


# %% ÉTAPE 4 : additionner les boîtes d'un même médicament (CIS), mois par mois
# Un médicament (CIS) peut exister en boîte de 30 et en boîte de 90 : on additionne.
# Puis on complète : si un mois manque pour un médicament, c'est 0 boîte.
ventes = medicam.groupby(["cis", "mois"], as_index=False)[["boites", "unites"]].sum()
grille = pd.MultiIndex.from_product([ventes.cis.unique(), sorted(ventes.mois.unique())], names=["cis", "mois"])
ventes = ventes.set_index(["cis", "mois"]).reindex(grille, fill_value=0).reset_index()
print("Ventes :", ventes.cis.nunique(), "médicaments sur", ventes.mois.nunique(), "mois")


# %% ÉTAPE 5 : construire la fiche de chaque médicament
fiches = spe.set_index("cis").reindex(ventes.cis.unique()).dropna(subset=["nom"])

# Code ATC : celui de la boîte la plus vendue du médicament.
total = medicam.groupby(["cis", "atc"]).boites.sum().reset_index().sort_values("boites", ascending=False)
fiches["atc"] = total.drop_duplicates("cis").set_index("cis").atc.reindex(fiches.index)

fiches["mitm"] = fiches.index.isin(mitm.cis)                       # intérêt thérapeutique majeur ?
fiches["vaccin"] = fiches.atc.fillna("").str.startswith("J07")     # les vaccins suivent des campagnes


# %% ÉTAPE 6 : regrouper les médicaments équivalents en "traitements"
# Si une marque manque, le pharmacien peut en donner une autre.
# Deux médicaments sont dans le même traitement s'ils ont :
#   - la même substance (ATC), la même dose et la même famille de forme, OU
#   - le même groupe générique, OU
#   - le même nom (un produit qui a juste changé de code).
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
    """'1 g' et '1000 mg' doivent être égaux."""
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

fiches["fam"] = fiches.forme.map(famille_forme)
c = compo[compo.cis.isin(fiches.index)].copy()
c["priorite"] = (c.nature == "FT").astype(int)                     # on préfère la forme active (FT)
c = c.sort_values(["cis", "lien", "priorite"], ascending=[True, True, False]).drop_duplicates(["cis", "lien"])
c["dose"] = c.dosage.map(dose_normalisee)
fiches["dose"] = c.groupby("cis").dose.apply(lambda s: "+".join(sorted(s)))

# Relier les médicaments : chaque lien "même ATC+dose+forme", "même nom" ou
# "même groupe générique" les met dans le même paquet (comme des amis d'amis).
parent = {}
def chef(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def relier(a, b):
    parent[chef(a)] = chef(b)

groupe_gen = gener[gener.type.isin(["0", "1", "4"])].drop_duplicates("cis").set_index("cis").groupe
for cis, r in fiches.iterrows():
    relier(f"I{cis}", f"K{r.atc}|{r.dose}|{r.fam}")
    relier(f"I{cis}", "S" + re.sub(r"[^a-z0-9%]", "", sans_accent(r.nom).split(",")[0]) + "|" + r.fam)
    if cis in groupe_gen.index:
        relier(f"I{cis}", f"G{groupe_gen[cis]}")
racine = pd.Series({cis: chef(f"I{cis}") for cis in fiches.index})
fiches["traitement"] = "T" + racine.groupby(racine).ngroup().astype(str).str.zfill(5)
print("Médicaments :", len(fiches), "-> traitements :", fiches.traitement.nunique())


# %% ÉTAPE 7 : repérer les médicaments arrêtés
# Un arrêt de commercialisation n'est pas une pénurie : on le marque pour l'écarter plus tard.
# Arrêté = toutes ses boîtes vendues sont déclarées arrêtées, ou le médicament n'est plus commercialisé.
p = pres[pres.cis.isin(fiches.index)].copy()
p["arret"] = p.etat_commercialisation.str.contains("arr", case=False, na=False)
p["vendu"] = p.cip13.map(medicam.groupby("cip13").boites.sum()).fillna(0) > 0
tout_arrete = p[p.vendu].groupby("cis").arret.all()
fiches["arrete"] = tout_arrete.reindex(fiches.index).fillna(False).astype(bool) | fiches.etat.ne("Commercialisée")
print("Arrêtés :", int(fiches.arrete.sum()), "| vaccins :", int(fiches.vaccin.sum()), "| MITM :", int(fiches.mitm.sum()))


# %% ÉTAPE 8 : signalements : garder ceux qu'on relie à un médicament, un seul par jour
signal = signal.dropna(subset=["cis"])
signal["date"] = pd.to_datetime(signal.date, format="mixed", dayfirst=True, errors="coerce")
signal = signal.dropna(subset=["date"]).drop_duplicates(["cis", "date"]).sort_values("date")
print("Signalements reliés :", len(signal), "du", signal.date.min().date(), "au", signal.date.max().date())


# %% ÉTAPE 9 : enregistrer les 3 fichiers propres
ventes = ventes[ventes.cis.isin(fiches.index)]
ventes.to_csv(f"{SORTIE}/ventes_par_medicament.csv", index=False)
signal[["date", "cip13", "cis"]].to_csv(f"{SORTIE}/signalements_propres.csv", index=False)
fiches.rename_axis("cis").reset_index()[
    ["cis", "nom", "titulaire", "atc", "fam", "dose", "traitement", "mitm", "vaccin", "arrete"]
].to_csv(f"{SORTIE}/fiches_medicaments.csv", index=False)
print("Écrit dans", SORTIE, ": ventes_par_medicament.csv, signalements_propres.csv, fiches_medicaments.csv")
