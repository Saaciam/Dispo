# =============================================================================
# AvailScore : prédire les signalements de rupture ou de risque de rupture
#
# Question : ce médicament sera-t-il signalé à l'ANSM dans les 3 prochains mois ?
#
# Fichiers d'entrée (sortis de 1_nettoyage_final.py, dans resultats/) :
#   ventes_par_medicament.csv    ventes mensuelles de chaque médicament
#   signalements_propres.csv     signalements ANSM, 2022 à 2025
#   fiches_medicaments.csv       fiche de chaque médicament
#
# Sortie : resultats/tableau_final.csv (le tableau du machine learning) et les notes affichées.
#
# Lancer :  python 2_machine_learning.py      (depuis le dossier python/)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

HORIZON = 3          # on prédit les 3 mois qui suivent
TOP = 100            # on regarde les 100 médicaments les plus suspects


# %% ÉTAPE 1 : charger les trois fichiers
ventes = pd.read_csv("../resultats/ventes_par_medicament.csv", dtype={"cis": str})
signal = pd.read_csv("../resultats/signalements_propres.csv", dtype={"cis": str}, parse_dates=["date"])
fiches = pd.read_csv("../resultats/fiches_medicaments.csv", dtype={"cis": str})


# %% ÉTAPE 2 : garder les médicaments qu'on peut suivre
# On garde les médicaments vendus au moins 100 boîtes par mois en moyenne,
# encore commercialisés et hors vaccins : sinon les ventes ne veulent rien dire.
volume = ventes.groupby("cis").boites.median()
fiches = fiches[fiches.cis.isin(volume[volume >= 100].index) & ~fiches.arrete & ~fiches.vaccin]
ventes = ventes[ventes.cis.isin(fiches.cis)]


# %% ÉTAPE 3 : mettre les ventes en tableau médicament × mois
# Une ligne par médicament, une colonne par mois : on lit le passé facilement.
tab = ventes.pivot(index="cis", columns="mois", values="unites").fillna(0)
mois = list(tab.columns)


# %% ÉTAPE 4 : compter les signalements de chaque médicament, mois par mois
# Plusieurs déclarations le même mois comptent pour un seul signalement.
signal["mois"] = signal.date.dt.strftime("%Y-%m")
sig = pd.crosstab(signal.cis, signal.mois).clip(upper=1)
tous_mois = sorted(set(sig.columns) | set(mois))
sig = sig.reindex(index=tab.index, columns=tous_mois, fill_value=0)


# %% ÉTAPE 5 : ajouter quelques informations fixes sur chaque médicament
# Nombre de laboratoires qui fabriquent le même traitement : s'il n'y en a
# qu'un, il n'y a pas de solution de rechange.
labos = fiches.groupby("traitement").titulaire.nunique()
fiches["un_seul_labo"] = (fiches.traitement.map(labos) == 1).astype(int)
fiches["classe"] = fiches.atc.str[:4]       # famille de médicaments (ATC niveau 3)
fiches = fiches.set_index("cis").reindex(tab.index)


# %% ÉTAPE 6 : construire UN seul tableau, une ligne par médicament et par mois
# Pour chaque mois t, on ne regarde QUE le passé (jusqu'à t),
# et la réponse à trouver est dans le futur (t+1 à t+3).
lignes = []
for t in range(6, len(mois)):                        # il faut au moins 6 mois de passé
    m = mois[t]
    j = tous_mois.index(m)
    if j + HORIZON >= len(tous_mois) or tous_mois[j + HORIZON] > "2025-12":
        break                                        # les signalements s'arrêtent fin 2025
    passe12 = tab.iloc[:, max(0, t - 12):t]             # jusqu'à 12 mois de passé
    normal = passe12.median(axis=1).replace(0, np.nan)
    d = pd.DataFrame(index=tab.index)
    d["mois"] = m
    # --- ce que disent les ventes
    d["ventes_mois"] = (tab.iloc[:, t] / normal).clip(upper=2)                      # ce mois-ci / mois normal
    d["ventes_3mois"] = (tab.iloc[:, t - 2:t + 1].mean(axis=1) / normal).clip(upper=2)
    d["ventes_instables"] = (passe12.std(axis=1) / passe12.mean(axis=1)).clip(upper=3)
    d["volume"] = np.log10(passe12.mean(axis=1) + 1)
    # --- ce que disent les signalements passés
    d["signal_3mois"] = sig.iloc[:, j - 2:j + 1].sum(axis=1)
    d["signal_12mois"] = sig.iloc[:, j - 11:j + 1].sum(axis=1)
    d["signal_24mois"] = sig.iloc[:, max(0, j - 23):j + 1].sum(axis=1)
    # --- tension dans la famille : part des médicaments de la classe signalés ce trimestre
    d["classe_signalee"] = (d.signal_3mois > 0).groupby(fiches.classe).transform("mean")
    # --- fiche du médicament
    d["mitm"] = fiches.mitm.astype(int)
    d["un_seul_labo"] = fiches.un_seul_labo
    d["injectable"] = (fiches.fam == "inj").astype(int)
    # --- LA RÉPONSE : signalé dans les 3 mois suivants ?
    d["cible"] = (sig.iloc[:, j + 1:j + 1 + HORIZON].sum(axis=1) > 0).astype(int)
    lignes.append(d.reset_index())

data = pd.concat(lignes, ignore_index=True).fillna(0)
data.to_csv("../resultats/tableau_final.csv", index=False)
print("Tableau final :", data.shape, "| mois", data.mois.min(), "à", data.mois.max(),
      "| part signalée :", round(data.cible.mean() * 100, 1), "%")

VARIABLES = ["ventes_mois", "ventes_3mois", "ventes_instables", "volume", "signal_3mois", "signal_12mois",
             "signal_24mois", "classe_signalee", "mitm", "un_seul_labo", "injectable"]


# %% ÉTAPE 7 : découper dans le temps en train / validation / test
# On découpe par date, jamais au hasard : on apprend sur le passé et on teste sur le futur.
# On laisse 2 mois vides entre les blocs, pour que les réponses (3 mois à l'avance)
# d'un bloc ne débordent pas sur le bloc suivant.
train = data[data.mois <= "2025-01"]
val = data[data.mois.between("2025-04", "2025-05")]
test = data[data.mois.between("2025-08", "2025-09")]
for nom, x in (("train", train), ("validation", val), ("test", test)):
    print(f"{nom:10s} {x.mois.min()} à {x.mois.max()} : {len(x)} lignes, {x.cible.mean() * 100:.1f} % signalés")


# %% ÉTAPE 8 : les versions de la formule à comparer
def v1_repeter(x):
    """Version 1 : un médicament déjà signalé le sera encore."""
    return x.signal_12mois

def v2_points(x):
    """Version 2 : formule à points, écrite à la main."""
    return (3 * (x.signal_3mois > 0) + 2 * (x.signal_12mois > 0) + 1 * (x.signal_24mois >= 3)
            + 1 * (x.ventes_3mois < 0.85) + 1 * (x.classe_signalee >= 0.1) + 1 * x.un_seul_labo)

def v3_logistique():
    """Version 3 : formule à points dont les poids sont appris sur les données."""
    # Les informations sont d'abord mises à la même échelle, pour que les poids soient comparables.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

def v4_ml():
    """Version 4 : machine learning (arbres de décision combinés)."""
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=50, random_state=0)


def evaluer(appr, eval_):
    """Entraîne chaque version sur 'appr', puis la note sur 'eval_'. Trois notes simples :
    - AUC : chance de mieux noter un médicament signalé qu'un non signalé (0,5 = hasard, 1 = parfait)
    - AP : qualité de la liste des plus suspects, de 0 à 1 (le hasard vaut la part de signalés)
    - top100 : parmi les 100 médicaments les plus suspects de chaque mois, part vraiment signalée"""
    res = {}
    modeles = {"V1 répéter le passé": v1_repeter, "V2 points à la main": v2_points,
               "V3 points appris": v3_logistique(), "V4 machine learning": v4_ml()}
    for nom, mod in modeles.items():
        if callable(mod) and not hasattr(mod, "fit"):
            score = mod(eval_)
        else:
            mod.fit(appr[VARIABLES], appr.cible)
            score = mod.predict_proba(eval_[VARIABLES])[:, 1]
        score = pd.Series(np.asarray(score, dtype=float), index=eval_.index)
        top = [eval_.cible[score[eval_.mois == m].nlargest(TOP).index].mean() for m in eval_.mois.unique()]
        res[nom] = {"AUC": round(roc_auc_score(eval_.cible, score), 3),
                    "AP": round(average_precision_score(eval_.cible, score), 3),
                    "top100": round(float(np.mean(top)), 3)}
    return pd.DataFrame(res).T


# %% ÉTAPE 9 : entraîner sur train, comparer sur la validation
print("\nVALIDATION (entraîné sur train) :")
print(evaluer(train, val))


# %% ÉTAPE 10 : validation croisée dans le temps
# On rejoue l'exercice plusieurs fois : à chaque tour, on apprend sur tout le passé
# disponible, on laisse 2 mois vides, puis on teste sur 2 mois. Le test final reste caché.
tours = [("2024-10", "2025-01", "2025-02"), ("2024-12", "2025-03", "2025-04"),
         ("2025-02", "2025-05", "2025-06")]
notes = []
for fin_appr, debut, fin in tours:
    appr = data[data.mois <= fin_appr]
    ev = data[data.mois.between(debut, fin)]
    n = evaluer(appr, ev)
    n["tour"] = f"apprend jusqu'à {fin_appr}, teste {debut}-{fin}"
    notes.append(n)
cv = pd.concat(notes)
print("\nVALIDATION CROISÉE, tour par tour :")
print(cv)
print("\nVALIDATION CROISÉE, moyenne des tours :")
print(cv.groupby(level=0)[["AUC", "AP", "top100"]].mean().round(3))


# %% ÉTAPE 11 : test final, une seule fois, sur des mois jamais utilisés
# On réentraîne sur train + validation, puis on note sur le test.
print("\nTEST FINAL (entraîné sur train + validation) :")
appr = data[data.mois <= "2025-05"]
print(evaluer(appr, test))


# %% ÉTAPE 12 : comprendre ce qui compte
# Version 3 : un poids positif augmente le risque, un poids négatif le diminue.
mod = v3_logistique().fit(appr[VARIABLES], appr.cible)
poids = pd.Series(mod[-1].coef_[0], index=VARIABLES).round(2).sort_values(ascending=False)
print("\nPoids de la formule apprise (version 3) :")
print(poids)

# Version 4 : on mélange une information au hasard et on regarde combien la note baisse.
# Plus elle baisse, plus l'information est utile.
ml = v4_ml().fit(appr[VARIABLES], appr.cible)
imp = permutation_importance(ml, test[VARIABLES], test.cible, scoring="roc_auc", n_repeats=5, random_state=0)
print("\nCe qui compte le plus pour le machine learning (baisse de l'AUC) :")
print(pd.Series(imp.importances_mean, index=VARIABLES).round(3).sort_values(ascending=False))


# %% ÉTAPE 13 : et pour les médicaments jamais signalés ?
# Prévoir qu'un médicament déjà signalé le sera encore est facile.
# Le vrai défi : repérer ceux qui n'ont rien eu depuis 12 mois.
neufs = test[test.signal_12mois == 0]
print(f"\nTEST, médicaments sans signalement depuis 12 mois ({len(neufs)} lignes, "
      f"{neufs.cible.mean() * 100:.1f} % signalés ensuite) :")
print(evaluer(appr, neufs).drop(index="V1 répéter le passé"))
