# =============================================================================
# AvailScore : l'application
#
# Un pharmacien cherche un médicament et choisit un mois. L'application répond :
#   1. AUJOURD'HUI : le médicament est-il en rupture de stock ou en tension
#      d'approvisionnement ? (statut publié par l'ANSM, revérifié automatiquement)
#   2. POUR LE MOIS CHOISI : quelle est la probabilité d'une déclaration de rupture
#      ou de risque de rupture, et quelle part pour chacun des deux ?
#   3. Les équivalents et leur situation.
#
# Les données se mettent à jour toutes seules (voir mise_a_jour.py).
# Lancer (depuis le dossier availscore_app) :  python -m streamlit run app.py
# =============================================================================

import os
import pickle
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import mise_a_jour
import moteur

ICI = os.path.dirname(os.path.abspath(__file__))
RESULTATS = os.path.join(ICI, "resultats")
EXPORTS = os.path.join(ICI, "exports_power_bi")
CACHE = os.path.join(ICI, "cache_import")
PROPRES = ["ventes_par_medicament.csv", "signalements_propres.csv", "fiches_medicaments.csv"]
PAGES = ["Rechercher un médicament", "Liste à surveiller", "Performance des modèles", "Évaluation des modèles",
         "Mise à jour des données"]
MOIS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
           "novembre", "décembre"]

st.set_page_config(page_title="AvailScore", layout="wide")


# ---------------------------------------------------------------- outils
def nb(x):
    """12345 -> '12 345'"""
    return f"{int(x):,}".replace(",", " ")


def points(n):
    return f"{int(n)} point" + ("s" if int(n) > 1 else "")


def pct(x, dec=1):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def mois_fr(m):
    return pd.Period(m, "M").strftime("%m/%Y")


def mois_long(m):
    p = pd.Period(m, "M")
    return f"{MOIS_FR[p.month - 1]} {p.year}"


def barre(texte):
    """Une barre de progression et la fonction qui la fait avancer."""
    b = st.progress(0.0, text=texte)
    return b, lambda f, m: b.progress(min(max(f, 0.0), 1.0), text=f"{m} — {f:.0%}")


def lancer_mise_a_jour(telecharger=False):
    """Refait toute la chaîne avec une barre de progression ; garde le rapport pour l'afficher."""
    b, avance = barre("Mise à jour des données…")
    try:
        rapport = mise_a_jour.tout_mettre_a_jour(avance, telecharger=telecharger)
    except Exception as e:  # on montre l'erreur, l'ancienne version des données reste utilisable
        b.empty()
        st.session_state["rapport"] = {"erreur": str(e)}
        return False
    b.empty()
    rapport["heure"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    st.session_state["rapport"] = rapport
    st.cache_resource.clear()
    return True


@st.cache_resource
def memoire():
    """Résultats gardés en mémoire pour toutes les sessions, tant que les données ne changent pas.
    (Pas de @st.cache_resource directement sur les calculs : il interdit d'y afficher une barre de progression.)"""
    return {}


def calcule_une_fois(nom, cle, texte, calcul):
    """Refait le calcul seulement si les données ont changé, avec une barre de progression."""
    m = memoire()
    if m.get(nom, (None,))[0] != cle:
        b, avance = barre(texte)
        m[nom] = (cle, calcul(avance))
        b.empty()
    return m[nom][1]


def evaluation_sur_disque(cle, avance):
    """La performance prend environ une minute : elle est gardée sur disque pour les lancements suivants."""
    chemin = os.path.join(CACHE, "performance.pkl")
    try:
        with open(chemin, "rb") as f:
            sauve = pickle.load(f)
        if sauve["cle"] == cle and "tests" in sauve["E"]:
            return sauve["E"]
    except Exception:
        pass
    E = moteur.evaluer(RESULTATS, P, avance)
    os.makedirs(CACHE, exist_ok=True)
    with open(chemin, "wb") as f:
        pickle.dump({"cle": cle, "E": E}, f)
    return E


@st.cache_data
def catalogue(_date):
    """Toutes les spécialités de la BDPM (même celles sans prévision) et la table CIP13 -> CIS."""
    spe = pd.read_csv(os.path.join(EXPORTS, "Specialites.csv"), dtype=str)
    pres = pd.read_csv(os.path.join(EXPORTS, "Presentations.csv"), dtype=str)
    return spe, pres.dropna(subset=["cip13"]).drop_duplicates("cip13").set_index("cip13").cis


# ------------------------------------------- traitement automatique au lancement
changes = mise_a_jour.changements()
manque = not all(os.path.exists(os.path.join(RESULTATS, f)) for f in PROPRES)
if manque or changes is None or any(changes.values()):
    if "auto_fait" not in st.session_state:          # une seule tentative par session
        st.session_state["auto_fait"] = True
        st.info("Des fichiers de données ont changé (ou n'ont jamais été traités) : "
                "traitement automatique en cours, l'application réapprendra ensuite le modèle.")
        lancer_mise_a_jour()
        st.rerun()
if manque:
    st.error("Les fichiers propres manquent dans resultats/ et le traitement automatique a échoué.")
    st.write(st.session_state.get("rapport", {}).get("erreur", ""))
    st.stop()

# Le statut actuel (rupture, tension) change tous les jours : revérifié au plus toutes les 12 heures.
if "statuts" not in st.session_state:
    with st.spinner("Vérification des statuts de disponibilité publiés par l'ANSM…"):
        st.session_state["statuts"] = mise_a_jour.rafraichir_statuts()
date_statuts = st.session_state["statuts"]


# ---------------------------------------------------------------- la prédiction
cle = (tuple(os.path.getmtime(os.path.join(RESULTATS, f)) for f in PROPRES + ["disponibilite_ansm.csv"]),
       datetime.now().strftime("%Y-%m"))
P = calcule_une_fois("prediction", cle, "Prédiction en cours…",
                     lambda av: moteur.preparer(RESULTATS, av, cache=os.path.join(CACHE, "prediction.pkl")))
S = P["situation"]
manifeste = mise_a_jour.lire_manifeste() or {}


# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.title("AvailScore")
    page = st.radio("Aller à", PAGES, label_visibility="collapsed")
    st.divider()
    if date_statuts:
        st.caption(f"**Statuts ANSM vérifiés le :** {date_statuts.strftime('%d/%m/%Y à %H:%M')}")
    st.caption(f"**Ventes Medic'AM connues jusqu'à :** {mois_fr(S['dernier_vente'])}")
    st.caption(f"**Déclarations ANSM connues jusqu'à :** {mois_fr(S['dernier_signal'])}")
    if manifeste.get("date_traitement"):
        st.caption("**Données traitées le :** "
                   + datetime.fromisoformat(manifeste["date_traitement"]).strftime("%d/%m/%Y à %H:%M"))
    if changes and any(changes.values()):
        st.warning("Des fichiers de donnees/ ont changé depuis le dernier traitement.")
    if st.button("Relancer la prédiction", width="stretch"):
        memoire().clear()
        for f in ("prediction.pkl", "performance.pkl"):
            if os.path.exists(os.path.join(CACHE, f)):
                os.remove(os.path.join(CACHE, f))
        st.rerun()

r_ = st.session_state.get("rapport")
if r_ and "erreur" in r_:
    st.error("La dernière mise à jour a échoué, l'application utilise les données précédentes.\n\n" + r_["erreur"])


def choisir_mois(cle_widget):
    """Le mois en cours et les 5 suivants ; le mois en cours par défaut."""
    choix = st.segmented_control(
        "Mois", S["mois"], default=S["maintenant"], key=cle_widget,
        format_func=lambda m: mois_long(m) + (" (en cours)" if m == S["maintenant"] else ""))
    return choix or S["maintenant"]


def carte_statut(statut_ligne, mitm):
    """Ce que l'ANSM publie aujourd'hui sur ce médicament."""
    if statut_ligne is None:
        st.success("**Aucune rupture ni tension publiée par l'ANSM aujourd'hui.**")
        if not mitm:
            st.caption("L'ANSM publie surtout les médicaments d'intérêt thérapeutique majeur (MITM). Ce médicament "
                       "n'en est pas un : l'absence de statut ne garantit pas qu'il soit disponible partout.")
        return
    s = statut_ligne
    lien = f" — [fiche ANSM]({s.lien})" if isinstance(s.lien, str) else ""
    maj = f" (mis à jour le {s.date_maj})" if isinstance(s.date_maj, str) else ""
    if s.statut == "Rupture de stock":
        st.error(f"**En rupture de stock** depuis le {s.date_debut}{maj}. "
                 + (f"Remise à disposition prévue le {s.date_remise}." if isinstance(s.date_remise, str) else
                    "Pas de date de remise à disposition annoncée.") + lien)
    elif s.statut == "Tension d'approvisionnement":
        st.warning(f"**Risque de rupture (tension d'approvisionnement)** depuis le {s.date_debut}{maj} : "
                   "le médicament est encore disponible, mais en quantité limitée." + lien)
    elif s.statut == "Remise à disposition":
        depuis = s.date_remise if isinstance(s.date_remise, str) else s.date_maj
        st.success(f"**De nouveau disponible** depuis le {depuis}, après une rupture ou "
                   f"une tension commencée le {s.date_debut}." + lien)
    else:
        st.info(f"**{s.statut}** depuis le {s.date_debut}{maj}." + lien)


# ================================================================ 1. RECHERCHER UN MÉDICAMENT
if page == PAGES[0]:
    st.title("Disponibilité d'un médicament")
    spe, cip_vers_cis = catalogue(manifeste.get("date_traitement"))
    noms = dict(zip(spe.cis, spe.nom))
    c1, c2 = st.columns([3, 1])
    choix = c1.selectbox("Médicament", sorted(noms, key=noms.get), index=None,
                         format_func=lambda c: f"{noms[c]} ({c})", placeholder="Tapez quelques lettres du nom…")
    cip = c2.text_input("ou code CIP13 de la boîte", max_chars=13).strip()
    if cip:
        if cip in cip_vers_cis.index:
            choix = cip_vers_cis[cip]
        else:
            c2.caption("Code CIP13 inconnu de la BDPM.")
    mois = choisir_mois("mois_recherche")

    if not choix:
        st.info(f"Choisissez un médicament pour voir s'il est en rupture aujourd'hui et son risque pour "
                f"{mois_long(mois)}.")
    else:
        b, avance = barre("Prédiction en cours…")
        avance(0.2, "Statut ANSM actuel")
        d = P["dispo"][P["dispo"].cis == choix]
        statut = d.iloc[0] if len(d) else None
        avance(0.5, f"Calcul du risque pour {mois_long(mois)}")
        M = moteur.pour_le_mois(P, mois)
        ligne = M[M.cis == choix]
        r = ligne.iloc[0] if len(ligne) else None
        avance(0.8, "Recherche des équivalents")
        fiche = P["tout"][P["tout"].cis == choix]
        trait = fiche.traitement.iloc[0] if len(fiche) else None
        equiv = P["tout"][(P["tout"].traitement == trait) & (P["tout"].cis != choix)] if trait else P["tout"].iloc[:0]
        avance(1, "Prédiction terminée")
        b.empty()

        st.subheader(noms.get(choix, choix))
        info = spe[spe.cis == choix].iloc[0]
        st.caption(f"CIS {choix} — {info.titulaire} — {info.etat}"
                   + (f" — ATC {fiche.atc.iloc[0]}" if len(fiche) and isinstance(fiche.atc.iloc[0], str) else ""))
        mitm = bool(fiche.mitm.iloc[0]) if len(fiche) else False

        # ---- 1. aujourd'hui
        st.markdown("#### Aujourd'hui")
        carte_statut(statut, mitm)

        # ---- 2. prévision
        st.markdown(f"#### Prévision pour {mois_long(mois)}")
        if r is None:
            raison = ("il n'est plus commercialisé" if len(fiche) and fiche.arrete.iloc[0] else
                      "c'est un vaccin (ventes liées aux campagnes)" if len(fiche) and fiche.vaccin.iloc[0] else
                      "ses ventes en ville sont trop faibles ou inconnues (moins de 100 boîtes par mois)")
            st.info(f"Pas de prévision pour ce médicament : {raison}. Le statut ANSM ci-dessus reste valable.")
        else:
            if statut is not None and statut.statut in ("Rupture de stock", "Tension d'approvisionnement"):
                st.warning(f"Ce médicament est **déjà en {statut.statut.lower()}** aujourd'hui. La prévision "
                           "ci-dessous ne peut pas en tenir compte : le modèle ne voit les déclarations qu'avec "
                           f"{S['retard']} mois de retard (listes ANSM publiées une fois par an). Tant que l'ANSM "
                           "n'annonce pas de remise à disposition, considérez-le comme indisponible ou à risque.")
            c = st.columns(4)
            c[0].metric(f"Déclaration à l'ANSM en {mois_long(mois)}", pct(r.risque),
                        help="Probabilité que le laboratoire déclare une rupture de stock OU un risque de rupture "
                             "concernant ce médicament pendant ce mois.")
            c[1].metric("dont rupture de stock", "≈ " + pct(r.risque_rupture),
                        help="Le médicament manque : la pharmacie ne peut pas le délivrer dans les 72 heures.")
            c[2].metric("dont risque de rupture", "≈ " + pct(r.risque_tension),
                        help="Le laboratoire prévoit des difficultés d'approvisionnement (tension).")
            c[3].metric("Niveau", r.niveau, f"× {r.fois_moyenne:.1f} la moyenne".replace(".", ","),
                        delta_color="inverse" if r.fois_moyenne >= 1 else "normal")
            st.write(f"En moyenne, {pct(r.moyenne)} des médicaments font l'objet d'une déclaration dans un mois "
                     f"donné. Pour ce médicament, le modèle estime ce risque à **{pct(r.risque)}**, "
                     f"soit **{f'{r.fois_moyenne:.1f}'.replace('.', ',')} fois la moyenne** "
                     f"(rang {r.rang} sur {len(M)} médicaments suivis).")
            st.write(f"**Signes repérés :** {moteur.raisons(r)}.")
            if pd.notna(r.n_ruptures):
                hist = f"{int(r.n_ruptures)} rupture(s) et {int(r.n_risques)} risque(s) de rupture déclarés"
            else:
                hist = "aucune déclaration"
            st.caption(f"Répartition rupture / risque de rupture : estimée d'après l'historique de ce médicament "
                       f"({hist} entre {P['periode_types'][0]} et {P['periode_types'][1]}) et la part nationale "
                       f"récente ({pct(P['part_rupture_nationale'], 0)} de ruptures). Elle est indicative : "
                       "les données ANSM ne permettent pas de la prédire précisément.")

            # risque des 6 mois proposés
            serie = pd.DataFrame({"mois": S["mois"],
                                  "Risque": [r[f"risque_{S['ecarts'][m]}"] for m in S["mois"]],
                                  "Moyenne": [P["moyennes"][S["ecarts"][m]] for m in S["mois"]]})
            serie["Mois"] = serie.mois.map(mois_long)
            serie["Choisi"] = serie.mois == mois
            base = alt.Chart(serie).encode(x=alt.X("Mois:N", sort=list(serie.Mois), title=None))
            g = (base.mark_bar().encode(
                    y=alt.Y("Risque:Q", axis=alt.Axis(format="%"), title="Probabilité de déclaration"),
                    color=alt.condition(alt.datum.Choisi, alt.value("#d95f02"), alt.value("#9ecae1")),
                    tooltip=["Mois", alt.Tooltip("Risque:Q", format=".1%"), alt.Tooltip("Moyenne:Q", format=".1%")])
                 + base.mark_tick(color="black", thickness=2, size=40).encode(y="Moyenne:Q"))
            st.write("**Le risque mois par mois** (trait noir : moyenne de tous les médicaments)")
            st.altair_chart(g, width="stretch")

        # ---- 3. équivalents
        if len(equiv):
            st.markdown(f"#### Équivalents (même substance, même dose, même forme) pour {mois_long(mois)}")
            e = equiv[["cis", "nom", "titulaire"]].merge(P["dispo"][["cis", "statut"]], on="cis", how="left") \
                .merge(M[["cis", "risque", "niveau"]], on="cis", how="left").sort_values("risque")
            st.dataframe(pd.DataFrame({
                "Médicament": e.nom, "Laboratoire": e.titulaire,
                "Statut ANSM aujourd'hui": e.statut.fillna("aucune alerte publiée"),
                f"Risque en {mois_long(mois)}": e.risque.map(lambda x: pct(x) if x == x else "pas de prévision"),
                "Niveau": e.niveau.fillna("—")}), hide_index=True, width="stretch")
            st.caption("Triés du plus sûr au plus à risque : utile pour choisir une substitution.")

        # ---- 4. historique
        with st.expander("Historique des ventes et des déclarations"):
            hv = P["ventes"][P["ventes"].cis == choix].assign(date=lambda x: pd.to_datetime(x.mois + "-01"))
            sg = P["signal"][P["signal"].cis == choix].copy()
            if len(hv):
                pts = sg.assign(date=sg.date.dt.to_period("M").dt.to_timestamp()).drop_duplicates("date") \
                    .merge(hv[["date", "boites"]], on="date")
                st.altair_chart(alt.Chart(hv).mark_line().encode(
                    x=alt.X("date:T", title=None), y=alt.Y("boites:Q", title="Boîtes remboursées par mois"))
                    + alt.Chart(pts).mark_point(color="red", size=90, filled=True).encode(x="date:T", y="boites:Q"),
                    width="stretch")
                st.caption("Points rouges : mois avec au moins une déclaration ANSM.")
            if len(sg):
                st.write(f"{len(sg)} déclaration(s) dans les listes ANSM, la dernière le "
                         f"{sg.date.max().strftime('%d/%m/%Y')}.")
            else:
                st.write("Aucune déclaration dans les listes ANSM depuis 2019.")


# ================================================================ 2. LISTE À SURVEILLER
elif page == PAGES[1]:
    st.title("Médicaments à surveiller")
    mois = choisir_mois("mois_liste")
    M = moteur.pour_le_mois(P, mois)
    st.write(f"Probabilité que le laboratoire déclare à l'ANSM une rupture de stock ou un risque de rupture "
             f"**pendant {mois_long(mois)}**. En moyenne : {pct(M.moyenne.iloc[0])} des médicaments par mois.")
    c = st.columns(4)
    c[0].metric("Médicaments suivis", f"{len(M):,}".replace(",", " "))
    c[1].metric("Risque élevé", int((M.niveau == "Élevé").sum()), help="Au moins 3 fois la moyenne")
    c[2].metric("À surveiller", int((M.niveau == "À surveiller").sum()), help="Au moins 1,5 fois la moyenne")
    c[3].metric("En rupture aujourd'hui (ANSM)", int((M.statut == "Rupture de stock").sum()))

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    texte = f1.text_input("Filtrer par nom, laboratoire ou code ATC", "")
    niveaux = f2.multiselect("Niveau", ["Élevé", "À surveiller", "Faible"], default=["Élevé", "À surveiller"])
    statuts = f3.multiselect("Statut ANSM aujourd'hui", ["Rupture de stock", "Tension d'approvisionnement",
                                                         "Remise à disposition", "aucune alerte"])
    seulement_mitm = f4.checkbox("MITM")
    V = M[M.niveau.isin(niveaux)]
    if statuts:
        V = V[V.statut.fillna("aucune alerte").isin(statuts)]
    if seulement_mitm:
        V = V[V.mitm == 1]
    if texte:
        t = texte.lower()
        V = V[V.nom.str.lower().str.contains(t, regex=False) | V.titulaire.str.lower().str.contains(t, regex=False)
              | V.atc.fillna("").str.lower().str.startswith(t)]
    vue = pd.DataFrame({"Rang": V.rang, "Médicament": V.nom, "Risque": (V.risque * 100).round(1),
                        "dont rupture (≈ %)": (V.risque_rupture * 100).round(1),
                        "dont risque de rupture (≈ %)": (V.risque_tension * 100).round(1),
                        "Niveau": V.niveau, "Statut ANSM aujourd'hui": V.statut.fillna("—"),
                        "Pourquoi": V.apply(moteur.raisons, axis=1), "Laboratoire": V.titulaire, "ATC": V.atc})
    st.dataframe(vue, hide_index=True, width="stretch", height=520,
                 column_config={"Risque": st.column_config.ProgressColumn(
                     f"Risque en {mois_long(mois)} (%)", min_value=0, max_value=max(30, float(vue.Risque.max() or 0)),
                     format="%.1f %%")})
    st.caption(f"{len(vue)} médicaments affichés. C'est une liste à surveiller, pas une certitude : voir la "
               "page « Performance des modèles ».")
    st.download_button("Télécharger la liste (CSV)", vue.to_csv(index=False).encode("utf-8-sig"),
                       f"availscore_{mois}.csv", "text/csv")


# ================================================================ 3. PERFORMANCE
elif page == PAGES[2]:
    st.title("Performance des modèles")
    E = calcule_une_fois("performance", cle, "Évaluation des modèles en cours (environ une minute)…",
                         lambda av: evaluation_sur_disque(cle, av))

    st.write(f"Chaque mois proposé a son propre modèle, testé **dans la même situation que la prévision réelle** : "
             f"déclarations ANSM vues avec {S['retard']} mois de retard, même nombre de mois entre les dernières "
             "ventes connues et le mois prédit. Le test porte sur les 2 derniers mois dont on connaît la réponse, "
             "jamais vus pendant l'apprentissage.")

    st.subheader("Fiabilité pour chaque mois proposé")
    pm = E["par_mois"]
    st.dataframe(pd.DataFrame({
        "Mois prédit": pm["Mois prédit"].map(mois_long),
        "Mois après les dernières ventes": pm["Écart (mois)"],
        "AUC": pm.AUC.map(lambda x: f"{x:.3f}".replace(".", ",")),
        "Top 100": pm["Top 100"].map(lambda x: pct(x, 0)),
        "Hasard": pm["Part déclarée"].map(pct),
        "Mois d'apprentissage": pm["Mois d'apprentissage"]}), hide_index=True, width="stretch")
    st.markdown("""
- **AUC** : on tire un médicament déclaré et un non déclaré ; c'est la chance que le modèle donne le plus gros
  risque au bon. 0,5 = pile ou face, 1 = parfait.
- **Top 100** : parmi les 100 médicaments les plus à risque, la part vraiment déclarée pendant le mois.
  À comparer avec le **hasard** (la part de médicaments déclarés dans un mois).
- Plus le mois est loin des dernières données, plus la prévision est difficile.""")

    v4 = E["notes"].loc["V4 machine learning"]
    st.subheader(f"Le détail pour {mois_long(S['maintenant'])} (mois en cours)")
    st.write(f"Apprentissage jusqu'à {mois_fr(E['fin_appr'])}, test sur "
             f"{' et '.join(mois_fr(m) for m in E['test_mois'])} ({nb(E['n_test'])} lignes, dont "
             f"{pct(E['part'])} déclarées).")
    c = st.columns(4)
    c[0].metric("AUC", f"{v4.AUC:.3f}".replace(".", ","))
    c[1].metric("AP", f"{v4.AP:.3f}".replace(".", ","), help=f"Le hasard vaut {E['part']:.3f}")
    c[2].metric("Top 100", pct(v4["Top 100"]), f"× {v4['Top 100'] / E['part']:.1f} le hasard".replace(".", ","))
    c[3].metric("AUC jamais déclarés", f"{v4['AUC jamais déclarés']:.3f}".replace(".", ","),
                help="Médicaments sans déclaration sur les 12 derniers mois connus : le cas le plus difficile")
    st.dataframe(E["notes"].style.format({"AUC": "{:.3f}", "AP": "{:.3f}", "Top 100": "{:.0%}",
                                          "AUC jamais déclarés": "{:.3f}"}, na_rep="—")
                 .highlight_max(color="#c7e9c0"), width="stretch")

    g1, g2 = st.columns(2)
    g1.write("**Les probabilités sont-elles justes ?**")
    fi = E["fiabilite"].reset_index()
    g1.dataframe(pd.DataFrame({"Niveau": fi.Niveau, "Médicaments": fi.Médicaments,
                               "Probabilité annoncée": fi.Risque_prédit.map(pct),
                               "Vraiment déclarés": fi.Vraiment_déclarés.map(pct)}), hide_index=True, width="stretch")
    g1.caption("Quand la probabilité annoncée et la part vraiment déclarée sont proches, on peut lire le "
               "pourcentage affiché comme une vraie probabilité.")
    g2.write("**Ce qui compte le plus pour le modèle**")
    g2.altair_chart(alt.Chart(E["importance"].head(10)).mark_bar(color="#1b9e77").encode(
        x=alt.X("Baisse de l'AUC:Q"), y=alt.Y("Variable:N", sort="-x", title=None),
        tooltip=["Variable", alt.Tooltip("Baisse de l'AUC:Q", format=".3f")]), width="stretch")

    st.write("**Précision selon la longueur de la liste**")
    st.altair_chart(alt.Chart(E["courbe"]).mark_line(point=True).encode(
        x=alt.X("Longueur de la liste:Q", scale=alt.Scale(type="log")),
        y=alt.Y("Part déclarée:Q", axis=alt.Axis(format="%"), title="Part vraiment déclarée"),
        color=alt.Color("Version:N", legend=alt.Legend(orient="bottom")),
        strokeDash=alt.condition(alt.datum.Version == "Hasard", alt.value([4, 4]), alt.value([1, 0])),
        tooltip=["Version", "Longueur de la liste", alt.Tooltip("Part déclarée:Q", format=".1%")]), width="stretch")

    st.subheader("Rupture ou risque de rupture ?")
    T = E["types"]
    if T:
        st.write(f"Test sur les {nb(T['n'])} déclarations de {T['annee']} : la part de ruptures estimée d'après "
                 f"l'historique de chaque médicament obtient une AUC de **{T['AUC']:.2f}**.".replace(".", ",")
                 + f" La part réelle était de {pct(T['part_reelle'], 0)}, pour "
                 f"{pct(T['part_prevue'], 0)} annoncés. La séparation entre rupture et risque de rupture est donc "
                 "**indicative** : elle repose surtout sur la tendance nationale.")
    else:
        st.write("Ajoutez le fichier Data.ansm « historique ruptures » dans donnees/ansm_historique/ "
                 "pour estimer la part de ruptures.")

    st.subheader("Les limites")
    st.markdown(f"""
- Les listes ANSM sont publiées une fois par an : les déclarations sont connues jusqu'à {mois_fr(S['dernier_signal'])}.
  Le modèle en tient compte, mais ce retard réduit sa précision. Chaque nouvelle liste l'améliore.
- Une déclaration dans un mois donné est un événement rare (environ 3 % des médicaments). Même un risque
  « élevé » reste souvent inférieur à 20 %.
- Les ventes Medic'AM ne couvrent que la ville (pas les médicaments délivrés à l'hôpital).
- Le statut « aujourd'hui » vient directement de l'ANSM. Il couvre surtout les médicaments d'intérêt
  thérapeutique majeur.""")


# ================================================================ 4. ÉVALUATION
elif page == PAGES[3]:
    st.title("Évaluation des modèles")
    E = calcule_une_fois("performance", cle, "Évaluation des modèles en cours (environ une minute)…",
                         lambda av: evaluation_sur_disque(cle, av))
    st.write("Ici, on teste les modèles **exemple par exemple** sur le jeu de test : des mois passés dont on "
             "connaît la réponse, mais que les modèles n'ont **jamais vus** pendant l'apprentissage. On compare "
             "ce que chaque version prédisait avec ce qui s'est vraiment passé, puis avec l'ensemble du jeu de test.")

    mois_choisi = choisir_mois("mois_eval")
    h = S["ecarts"][mois_choisi]
    T = E["tests"][h]
    jeu, moyenne = T["jeu"], T["moyenne"]
    noms_tout = dict(zip(P["tout"].cis, P["tout"].nom))
    st.caption(f"Modèle utilisé pour prédire {mois_long(mois_choisi)} : il prédit {h} mois après les dernières "
               f"ventes connues, avec des déclarations vues {S['retard']} mois en retard. Pour le test, il a appris "
               f"jusqu'à {mois_fr(T['fin_appr'])}, puis il a prédit "
               f"{' et '.join(mois_long(plus) for plus in sorted(jeu.mois_predit.unique()))} "
               f"({nb(len(jeu))} exemples, dont {pct(jeu.cible.mean())} vraiment déclarés).")

    # ---------------- résultats sur tout le jeu de test
    st.subheader("1. Les résultats sur tout le jeu de test")
    mat = moteur.matrice(jeu, moyenne)
    st.dataframe(mat.style.format({"Alertes justes (précision)": "{:.1%}", "Déclarations repérées (rappel)": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Une version « donne l'alerte » quand : V1, le médicament a été déclaré sur les 12 derniers mois "
               f"connus ; V2, il a au moins 3 points ; V3 et V4, sa probabilité atteint 1,5 fois la moyenne "
               f"({pct(1.5 * moyenne)}), c'est-à-dire le niveau « À surveiller ». La **précision** est la part des "
               "alertes qui étaient justifiées ; le **rappel** est la part des déclarations qui ont été repérées.")

    # ---------------- choisir un exemple
    st.subheader("2. Tester un exemple")
    cle_m, cle_c = f"ex_mois_{h}", f"ex_cis_{h}"
    mois_dispo = sorted(jeu.mois_predit.unique())
    al4 = moteur.alertes(jeu, moyenne)["v4"]

    def tirer(filtre):
        """Choisit un exemple au hasard parmi ceux qui respectent le filtre."""
        candidats = jeu[filtre]
        if len(candidats):
            x = candidats.sample(1).iloc[0]
            st.session_state[cle_m], st.session_state[cle_c] = x.mois_predit, x.cis

    b = st.columns(5)
    b[0].button("Au hasard", on_click=tirer, args=(jeu.cible >= 0,), width="stretch")
    b[1].button("Un médicament déclaré", on_click=tirer, args=(jeu.cible == 1,), width="stretch")
    b[2].button("Un non déclaré", on_click=tirer, args=(jeu.cible == 0,), width="stretch")
    b[3].button("Une fausse alerte V4", on_click=tirer, args=(al4 & (jeu.cible == 0),), width="stretch")
    b[4].button("Une déclaration manquée V4", on_click=tirer, args=(~al4 & (jeu.cible == 1),), width="stretch")

    c1, c2 = st.columns([1, 3])
    if st.session_state.get(cle_m) not in mois_dispo:
        st.session_state[cle_m] = mois_dispo[-1]
    m_ex = c1.selectbox("Mois prédit", mois_dispo, key=cle_m, format_func=mois_long)
    du_mois = jeu[jeu.mois_predit == m_ex]
    options = sorted(du_mois.cis, key=lambda c: noms_tout.get(c, c))
    if st.session_state.get(cle_c) not in options:
        st.session_state[cle_c] = du_mois.sort_values("v4", ascending=False).cis.iloc[0]
    cis_ex = c2.selectbox("Médicament", options, key=cle_c, format_func=lambda c: f"{noms_tout.get(c, c)} ({c})")
    ex = du_mois[du_mois.cis == cis_ex].iloc[0]
    alertes_ex = {v: bool(a[ex.name]) for v, a in moteur.alertes(jeu, moyenne).items()}

    # ce qui s'est vraiment passé
    st.markdown(f"**{noms_tout.get(cis_ex, cis_ex)}** — situation à la fin de {mois_long(ex.mois)}, "
                f"prédiction pour **{mois_long(m_ex)}**")
    decl = P["signal"][(P["signal"].cis == cis_ex) & (P["signal"].date.dt.strftime("%Y-%m") == m_ex)]
    if ex.cible == 1:
        detail = ", ".join(sorted(decl.date.dt.strftime("%d/%m/%Y").unique()))
        st.error(f"**Réalité : déclaré à l'ANSM en {mois_long(m_ex)}** ({len(decl)} déclaration(s), le {detail}).")
    else:
        st.success(f"**Réalité : aucune déclaration à l'ANSM en {mois_long(m_ex)}.**")

    # ce que chaque version prédisait
    rang = {v: int((du_mois[v] > ex[v]).sum()) + 1 for v in ["v1", "v2", "v3", "v4"]}
    valeur = {"v1": f"{int(ex.v1)} mois déclaré(s) sur 12", "v2": points(ex.v2),
              "v3": pct(ex.v3), "v4": pct(ex.v4)}
    st.dataframe(pd.DataFrame({
        "Version": [moteur.VERSIONS[v] for v in valeur],
        "Prédiction": list(valeur.values()),
        "Rang dans le mois": [f"{rang[v]} sur {len(du_mois)}" for v in valeur],
        "Alerte ?": ["oui" if alertes_ex[v] else "non" for v in valeur],
        "Verdict": [moteur.verdict(alertes_ex[v], ex.cible == 1) for v in valeur]}),
        hide_index=True, width="stretch")

    # comparaison avec le jeu de test
    tranches = pd.qcut(jeu.v4.rank(method="first"), 10, labels=False)
    par_tranche = (jeu.assign(tranche=tranches).groupby("tranche")
                   .agg(Risque_min=("v4", "min"), Risque_max=("v4", "max"), Prédit=("v4", "mean"),
                        Réel=("cible", "mean"), Exemples=("cible", "size")).reset_index())
    t_ex = int(tranches[ex.name])
    ligne_t = par_tranche.iloc[t_ex]
    st.write(f"**Comparaison avec le jeu de test :** la V4 donnait {pct(ex.v4)}. Parmi les "
             f"{nb(ligne_t.Exemples)} exemples du jeu de test dont le risque V4 était proche "
             f"(entre {pct(ligne_t.Risque_min)} et {pct(ligne_t.Risque_max)}), **{pct(ligne_t.Réel)} ont vraiment "
             f"été déclarés**, pour {pct(ligne_t.Prédit)} annoncés en moyenne.")
    par_tranche["Tranche"] = [f"{i + 1}" for i in par_tranche.tranche]
    par_tranche["Exemple"] = par_tranche.tranche == t_ex
    longue = par_tranche.melt(id_vars=["Tranche", "Exemple"], value_vars=["Prédit", "Réel"],
                              var_name="Mesure", value_name="Part")
    st.altair_chart(alt.Chart(longue).mark_bar().encode(
        x=alt.X("Tranche:N", title="Tranches de risque V4, de la plus faible (1) à la plus forte (10)"),
        xOffset="Mesure:N",
        y=alt.Y("Part:Q", axis=alt.Axis(format="%"), title="Part déclarée"),
        color=alt.Color("Mesure:N", scale=alt.Scale(domain=["Prédit", "Réel"], range=["#9ecae1", "#d95f02"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        opacity=alt.condition(alt.datum.Exemple, alt.value(1), alt.value(0.45)),
        tooltip=["Tranche", "Mesure", alt.Tooltip("Part:Q", format=".1%")]), width="stretch")
    st.caption(f"La tranche de l'exemple (n° {t_ex + 1}) est en couleur pleine. Si les barres « Prédit » et "
               "« Réel » se ressemblent, les probabilités du modèle sont fiables.")

    # ---------------- modifier l'exemple à la main
    st.subheader("3. Modifier l'exemple à la main")
    st.write("Changez les informations de l'exemple pour voir comment la prédiction réagit. Les valeurs de départ "
             "sont celles de l'exemple choisi.")
    with st.form(f"simulation_{h}_{cis_ex}_{m_ex}"):
        col = st.columns(3)
        v = {}
        v["ventes_mois"] = col[0].number_input("Ventes du mois / mois normal", 0.0, 2.0, float(ex.ventes_mois), 0.05)
        v["ventes_3mois"] = col[0].number_input("Ventes des 3 derniers mois / normal", 0.0, 2.0, float(ex.ventes_3mois), 0.05)
        v["tendance_6mois"] = col[0].number_input("Tendance sur 6 mois (-2 à 2)", -2.0, 2.0, float(ex.tendance_6mois), 0.05)
        v["ventes_instables"] = col[0].number_input("Ventes irrégulières (0 = très régulières)", 0.0, 3.0,
                                                    float(ex.ventes_instables), 0.05)
        boites = col[0].number_input("Unités vendues par mois (moyenne)", 0, 10**9, int(round(10 ** ex.volume - 1)), 1000)
        v["signal_3mois"] = col[1].number_input("Mois déclarés sur les 3 derniers connus", 0, 3, int(ex.signal_3mois))
        v["signal_12mois"] = col[1].number_input("Mois déclarés sur les 12 derniers connus", 0, 12, int(ex.signal_12mois))
        v["signal_24mois"] = col[1].number_input("Mois déclarés sur les 24 derniers connus", 0, 24, int(ex.signal_24mois))
        nb_decl = col[1].number_input("Nombre de déclarations sur 12 mois", 0, 500,
                                      int(round(np.expm1(ex.nb_declarations_12mois))))
        v["mois_depuis_declaration"] = col[1].number_input("Mois depuis la dernière déclaration (60 = jamais)", 0, 60,
                                                           int(ex.mois_depuis_declaration))
        v["classe_signalee"] = col[2].number_input("Part de la famille ATC déclarée (0 à 1)", 0.0, 1.0,
                                                   float(ex.classe_signalee), 0.05)
        v["equivalents_declares"] = col[2].number_input("Part des équivalents déclarés (0 à 1)", 0.0, 1.0,
                                                        float(ex.equivalents_declares), 0.05)
        v["ventes_traitement_3mois"] = col[2].number_input("Ventes de tous les équivalents / normal", 0.0, 2.0,
                                                           float(ex.ventes_traitement_3mois), 0.05)
        v["part_marche"] = col[2].number_input("Part des ventes parmi les équivalents (0 à 1)", 0.0, 1.0,
                                               float(min(1.0, ex.part_marche)), 0.05)
        v["n_labos"] = col[2].number_input("Nombre de laboratoires", 1, 10, int(max(1, ex.n_labos)))
        v["n_equivalents"] = col[2].number_input("Nombre d'équivalents", 1, 30, int(max(1, ex.n_equivalents)))
        v["mitm"] = int(col[2].checkbox("Intérêt thérapeutique majeur (MITM)", bool(ex.mitm)))
        v["injectable"] = int(col[2].checkbox("Injectable", bool(ex.injectable)))
        st.form_submit_button("Recalculer la prédiction", type="primary")
    incoherences = []
    if not v["signal_3mois"] <= v["signal_12mois"] <= v["signal_24mois"]:
        incoherences.append("les mois déclarés doivent croître : 3 derniers ≤ 12 derniers ≤ 24 derniers")
    derniere = v["mois_depuis_declaration"] - S["retard"]          # en mois avant le dernier mois connu
    if v["signal_3mois"] > 0 and derniere > 2:
        incoherences.append(f"déclaré sur les 3 derniers mois connus, donc « mois depuis la dernière déclaration » "
                            f"devrait valoir au plus {S['retard'] + 2}")
    if v["signal_24mois"] == 0 and v["mois_depuis_declaration"] < 60 and derniere <= 23:
        incoherences.append("aucun mois déclaré sur 24, donc « mois depuis la dernière déclaration » devrait dépasser "
                            f"{S['retard'] + 23} (60 = jamais)")
    if incoherences:
        st.warning("Valeurs incohérentes entre elles, la prédiction sera peu fiable : " + " ; ".join(incoherences) + ".")
    v["volume"] = float(np.log10(boites + 1))
    v["nb_declarations_12mois"] = float(np.log1p(nb_decl))
    v["un_seul_labo"] = int(v["n_labos"] == 1)
    sim = pd.DataFrame([v])[moteur.VARIABLES]
    p3, p4 = T["v3"].predict_proba(sim)[:, 1][0], T["v4"].predict_proba(sim)[:, 1][0]
    pts = int(moteur.v2_points(sim).iloc[0])
    c = st.columns(4)
    c[0].metric("V1 répéter le passé", f"{v['signal_12mois']} mois", f"{v['signal_12mois'] - int(ex.v1)} par rapport à l'exemple")
    c[1].metric("V2 points à la main", points(pts), f"{pts - int(ex.v2)} par rapport à l'exemple")
    c[2].metric("V3 points appris", pct(p3), f"{(p3 - ex.v3) * 100:+.1f} points".replace(".", ","), delta_color="inverse")
    c[3].metric("V4 machine learning", pct(p4), f"{(p4 - ex.v4) * 100:+.1f} points".replace(".", ","), delta_color="inverse")
    t_sim = par_tranche[(par_tranche.Risque_min <= p4)].tranche.max()
    t_sim = 0 if pd.isna(t_sim) else int(t_sim)
    ls = par_tranche.iloc[t_sim]
    st.write(f"Avec ces valeurs, la V4 annonce **{pct(p4)}** (niveau **{moteur.niveau([p4], moyenne)[0]}**, "
             f"{f'{p4 / moyenne:.1f}'.replace('.', ',')} fois la moyenne). Dans le jeu de test, les exemples de la "
             f"même tranche de risque (n° {t_sim + 1}) ont vraiment été déclarés dans **{pct(ls.Réel)}** des cas.")


# ================================================================ 5. MISE À JOUR
else:
    st.title("Mise à jour des données")
    st.write("Le traitement est **automatique** : au lancement, l'application compare les fichiers de "
             "`donnees/` à ceux du dernier traitement. Si un fichier a été ajouté, remplacé ou supprimé, elle "
             "relit les fichiers bruts, **vérifie leur format**, refait les 7 tables, relance le nettoyage, "
             "puis réapprend les modèles. Un fichier au format inattendu arrête tout avant d'écraser les données. "
             "Le statut de disponibilité publié par l'ANSM est revérifié toutes les 12 heures.")

    rapport = st.session_state.get("rapport")
    if rapport and "erreur" not in rapport:
        st.success(f"Dernière mise à jour réussie le {rapport['heure']}.")
        if rapport.get("telecharges") is not None:
            st.write("Fichiers téléchargés et modifiés : "
                     + (", ".join(rapport["telecharges"]) if rapport["telecharges"] else "aucun (déjà à jour)"))
        for a in rapport.get("avertissements", []):
            st.caption(a)
        av, ap = rapport.get("avant", {}), rapport.get("apres", {})
        lignes = [{"Indicateur": k, "Avant": str(av.get(k, "—")), "Après": str(ap.get(k, "—")),
                   "Changé": "oui" if str(av.get(k)) != str(ap.get(k)) else ""} for k in dict.fromkeys([*av, *ap])]
        st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch")
        with st.expander("Journal du nettoyage"):
            st.code(rapport.get("journal", ""))

    st.subheader("1. Télécharger les dernières données officielles")
    st.write("BDPM (base-donnees-publique.medicaments.gouv.fr), nouvelles listes ANSM des ruptures et risques de "
             "rupture (ansm.sante.fr), nouveaux semestres Medic'AM (data.gouv.fr). Chaque fichier est vérifié, "
             "puis tout le traitement est refait.")
    if st.button("Télécharger et mettre à jour", type="primary"):
        if lancer_mise_a_jour(telecharger=True):
            st.rerun()
        st.error(st.session_state["rapport"]["erreur"])

    st.subheader("2. Ajouter des fichiers à la main")
    st.write("Un fichier qui porte le même nom qu'un fichier existant le remplace.")
    u = st.columns(4)
    depots = {"ansm": u[0].file_uploader("Listes ANSM (.xlsx)", type=["xlsx", "xls"], accept_multiple_files=True),
              "medicam": u[1].file_uploader("Medic'AM (.xls, .xlsx)", type=["xls", "xlsx"], accept_multiple_files=True),
              "bdpm": u[2].file_uploader("Fichiers BDPM (.txt)", type=["txt"], accept_multiple_files=True),
              "ansm_historique": u[3].file_uploader("Data.ansm historique ruptures (.xlsx)", type=["xlsx"],
                                                    accept_multiple_files=True)}
    a_traiter = sum(len(v or []) for v in depots.values())
    if a_traiter and st.button(f"Enregistrer {a_traiter} fichier(s) et mettre à jour", type="primary"):
        for dossier, fichiers in depots.items():
            for f in fichiers or []:
                nom = os.path.basename(f.name)
                if dossier == "bdpm" and nom not in mise_a_jour.BDPM:
                    st.warning(f"{nom} ignoré : les fichiers BDPM attendus sont {', '.join(mise_a_jour.BDPM)}.")
                    continue
                os.makedirs(os.path.join(mise_a_jour.DONNEES, dossier), exist_ok=True)
                with open(os.path.join(mise_a_jour.DONNEES, dossier, nom), "wb") as out:
                    out.write(f.getbuffer())
        if lancer_mise_a_jour():
            st.rerun()
        st.error(st.session_state["rapport"]["erreur"])

    st.subheader("3. Relancer le traitement")
    if st.button("Refaire tout le traitement avec les fichiers actuels"):
        if lancer_mise_a_jour():
            st.rerun()
        st.error(st.session_state["rapport"]["erreur"])

    st.subheader("Fichiers utilisés")
    ch = mise_a_jour.changements() or {}
    etat = {f: "ajouté depuis le dernier traitement" for f in ch.get("ajoutés", [])}
    etat.update({f: "modifié depuis le dernier traitement" for f in ch.get("modifiés", [])})
    st.dataframe(pd.DataFrame([{"Fichier": f, "Taille (Ko)": round(t / 1024),
                                "Modifié le": datetime.fromtimestamp(d).strftime("%d/%m/%Y %H:%M"),
                                "État": etat.get(f, "traité")}
                               for f, (t, d) in mise_a_jour.empreinte().items()]),
                 hide_index=True, width="stretch")
    if ch.get("supprimés"):
        st.caption("Supprimés depuis le dernier traitement : " + ", ".join(ch["supprimés"]))
