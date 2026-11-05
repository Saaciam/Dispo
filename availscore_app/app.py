"""AvailScore : interface simple. Lancer avec :  streamlit run app.py"""
import datetime as dt
import json
import os
import shutil

import altair as alt
import pandas as pd
import streamlit as st

import pipeline

ICI = os.path.dirname(os.path.abspath(__file__))
DONNEES, RESULTATS, TRANSFORMEES = f"{ICI}/donnees", f"{ICI}/resultats", f"{ICI}/donnees_transformees"
SEUIL_TENDU, SEUIL_MANQUE, FACTEUR = pipeline.SEUIL_TENDU, pipeline.SEUIL_MANQUE, pipeline.FACTEUR_MENSUEL
MOIS_COURTS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

st.set_page_config(page_title="AvailScore", page_icon="💊", layout="centered")


@st.cache_data(show_spinner="Chargement des résultats...")
def charger():
    T = pd.read_csv(f"{RESULTATS}/traitements.csv.gz", dtype={"traitement": str})
    M = pd.read_csv(f"{RESULTATS}/medicaments.csv.gz", dtype={"cis": str, "traitement": str},
                    keep_default_na=False, na_values=[""])
    S = pd.read_csv(f"{RESULTATS}/series.csv.gz", dtype={"id": str})
    infos = json.load(open(f"{RESULTATS}/infos.json", encoding="utf-8"))
    return T, M, S, infos


def prevoir(dispo, h):
    """Disponibilite prevue dans h mois = 100 % - manque d'aujourd'hui x 0,9^h"""
    return 1 - (1 - dispo) * FACTEUR ** h


def niveau(d):
    return "Disponible" if d >= SEUIL_TENDU else "Tendu" if d >= SEUIL_MANQUE else "Manque"


def pct(d):
    return f"{d * 100:.1f} %".replace(".", ",")


def nombre(x):
    return f"{x:,.0f}".replace(",", " ")


def nom_mois(aaaa_mm, ajout=0):
    a, m = int(aaaa_mm[:4]), int(aaaa_mm[5:7]) - 1 + ajout
    return f"{MOIS[m % 12]} {a + m // 12}"


def etiquette_futur(aaaa_mm, ajout):
    a, m = int(aaaa_mm[:4]), int(aaaa_mm[5:7]) - 1 + ajout
    return f"{MOIS_COURTS[m % 12]} {str(a + m // 12)[2:]}"


def risque(points):
    return "élevé" if points >= 4 else "moyen" if points >= 2 else "faible"


def horizon_par_defaut(dernier):
    """Nombre de mois entre le dernier mois connu et aujourd'hui."""
    a, m = int(dernier[:4]), int(dernier[5:7])
    auj = dt.date.today()
    return min(12, max(1, (auj.year - a) * 12 + auj.month - m))


def etiquette(aaaa_mm):
    return f"{MOIS_COURTS[int(aaaa_mm[5:7]) - 1]} {aaaa_mm[2:4]}"


def graphique(serie, unite):
    d = serie[["mois", "reel", "attendu"]].rename(columns={"reel": "Délivré", "attendu": "Attendu"})
    d = d.melt("mois", var_name="Courbe", value_name="Quantité").dropna()
    d["Mois"] = d.mois.map(etiquette)
    ordre = [etiquette(m) for m in serie.mois]
    espace = "replace(format(datum.value, ',d'), ',', ' ')"
    return (alt.Chart(d).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Mois:N", title=None, sort=ordre, axis=alt.Axis(values=ordre[::-3][::-1], labelAngle=0, grid=False)),
        y=alt.Y("Quantité:Q", title=f"{unite} par mois", scale=alt.Scale(zero=True), axis=alt.Axis(labelExpr=espace)),
        color=alt.Color("Courbe:N", title=None, scale=alt.Scale(domain=["Délivré", "Attendu"], range=["#2563eb", "#9ca3af"]),
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("Courbe:N", scale=alt.Scale(domain=["Délivré", "Attendu"], range=[[1, 0], [5, 4]]), legend=None),
        tooltip=["Mois:N", "Courbe:N", alt.Tooltip("Quantité:Q", format=",.0f")],
    ).properties(height=280))


def calcul_detaille(ligne, h, unite):
    manque = 1 - min(ligne.dispo, 1)
    return pd.DataFrame([
        ("Niveau habituel", "Médiane des 12 derniers mois, gelée pendant un manque", f"{nombre(ligne.habituel)} {unite}"),
        ("Saison", "Même période l'an dernier ÷ niveau de l'an dernier", f"× {ligne.saison:.3f}".replace(".", ",")),
        ("Effet du mois", "Évolution des médicaments comparables sans alerte", f"× {ligne.effet:.3f}".replace(".", ",")),
        ("Attendu", "Niveau habituel × saison × effet du mois", f"{nombre(ligne.attendu)} {unite}"),
        ("Délivré", "Lu dans Medic'AM", f"{nombre(ligne.reel)} {unite}"),
        ("Disponibilité", "Délivré ÷ attendu, plafonnée à 100 %", pct(min(ligne.dispo, 1))),
        ("Manque", "100 % − disponibilité", pct(manque)),
        (f"Prévision à {h} mois", f"100 % − manque × 0,9^{h}", pct(1 - manque * FACTEUR ** h)),
    ], columns=["Étape", "Calcul", "Résultat"])


def page_medicament(T, M, S, infos):
    dernier = infos["dernier_mois"]
    noms = M.sort_values("nom")
    choix = st.selectbox("Quel médicament veux-tu tester ?", noms.index, index=None,
                         format_func=lambda i: M.nom[i], placeholder="Tape un nom, par exemple Clomid ou Doliprane")
    if choix is None:
        st.caption("La disponibilité compare ce qui a été délivré à ce qui aurait dû l'être.")
        return
    med = M.loc[choix]
    st.subheader(med.nom)
    st.caption(f"{med.titulaire} · Statut ANSM : {med.statut_ansm or 'aucun'}")
    if not med.traitement_calcule:
        st.info(f"**Calcul impossible.** {med.raison_hors_champ}")
        return

    h = st.slider("Prévoir dans combien de mois ?", 1, 12, horizon_par_defaut(dernier),
                  help=f"Compté à partir de {nom_mois(dernier)}, le dernier mois connu.")
    t = T[T.traitement == med.traitement].iloc[0]
    prevu = prevoir(t.dispo, h)
    etat = niveau(prevu)
    phrase = f"**{etat}** en {nom_mois(dernier, h)} : traitement prévu à {pct(prevu)}."
    {"Disponible": st.success, "Tendu": st.warning, "Manque": st.error}[etat](phrase)

    boite = med.dispo_boite if med.boite_calculee else None
    a, b, c = st.columns(3)
    a.metric(f"Cette boîte, {nom_mois(dernier)}", pct(boite) if boite is not None else "Non calculée",
             help="Boîtes délivrées ÷ boîtes attendues, pour cette spécialité.")
    b.metric(f"Le traitement, {nom_mois(dernier)}", pct(t.dispo), help="Même calcul, toutes marques équivalentes réunies.")
    c.metric(f"Traitement prévu à {h} mois", pct(prevu))

    if boite is not None and boite < SEUIL_MANQUE and t.dispo >= SEUIL_TENDU:
        st.info("Cette boîte manque, mais les autres marques du même traitement compensent.")
    if boite is None:
        st.caption(f"Cette boîte vend moins de {nombre(pipeline.VOLUME_MIN)} boîtes par mois : seul le traitement est calculé.")

    if etat == "Disponible":
        signes = [txt for ok, txt in [(t.signe_alerte, "une alerte ANSM couvre au moins la moitié du volume"),
                                      (t.signe_classe, "au moins 10 % des traitements de sa classe sont déjà en manque"),
                                      (t.signe_seul, "un seul laboratoire fournit presque tout"),
                                      (t.signe_non_mitm, "il n'est pas classé MITM")] if ok]
        st.write(f"**Risque de chute dans les 6 mois : {risque(t.points)}** ({t.points} point{'s' if t.points > 1 else ''})")
        if signes:
            st.caption("Signes observés : " + " ; ".join(signes) + ".")

    voir = "boite" if boite is not None else "traitement"
    ident = med.cis if voir == "boite" else med.traitement
    serie = S[(S.niveau == voir) & (S.id == ident)].sort_values("mois")
    unite = "boîtes" if voir == "boite" else "unités"
    st.markdown(f"#### {'Cette boîte' if voir == 'boite' else 'Le traitement'} : délivré et attendu")
    st.altair_chart(graphique(serie, unite.capitalize()), width="stretch")

    with st.expander("Voir le calcul pas à pas"):
        st.dataframe(calcul_detaille(serie.iloc[-1], h, unite), hide_index=True, width="stretch")
        if voir == "boite":
            st.caption("Les chiffres sont arrondis et ramenés en boîtes moyennes. La prévision affichée en haut porte sur le traitement.")

    autres = M[(M.traitement == med.traitement) & (M.index != choix)]
    if len(autres):
        with st.expander(f"Voir les {len(autres)} autres médicaments de ce traitement"):
            vue = autres.sort_values("boites_par_mois", ascending=False)
            st.dataframe(pd.DataFrame({"Médicament": vue.nom, "Boîtes/mois": vue.boites_par_mois,
                                       "Dispo": (vue.dispo_boite.where(vue.boite_calculee) * 100).round(1),
                                       "Statut ANSM": vue.statut_ansm}),
                         hide_index=True, width="stretch", column_config={
                             "Médicament": st.column_config.TextColumn(width=300),
                             "Boîtes/mois": st.column_config.NumberColumn(format="%d", width=100),
                             "Dispo": st.column_config.NumberColumn(format="%.1f %%", width=80)})


def page_liste(T, infos):
    dernier = infos["dernier_mois"]
    h = st.slider("Prévoir dans combien de mois ? ", 1, 12, horizon_par_defaut(dernier))
    vue = T.assign(prevu=prevoir(T.dispo, h))
    vue["niveau"] = vue.prevu.map(niveau)
    a, b, c = st.columns(3)
    a.metric("Disponibles", nombre((vue.niveau == "Disponible").sum()))
    b.metric("Tendus", nombre((vue.niveau == "Tendu").sum()))
    c.metric("En manque", nombre((vue.niveau == "Manque").sum()))
    choix = st.segmented_control("Afficher", ["Manque", "Tendu", "Disponible"], default="Manque")
    if choix:
        vue = vue[vue.niveau == choix]
    vue = vue.sort_values("boites_par_mois", ascending=False)
    dispo_nom = f"Dispo {etiquette(dernier)}"
    prevu_nom = f"Prévu {etiquette_futur(dernier, h)}"
    table = pd.DataFrame({"Traitement": vue.medicament_principal, dispo_nom: (vue.dispo * 100).round(1),
                          prevu_nom: (vue.prevu * 100).round(1), "Boîtes/mois": vue.boites_par_mois,
                          "Risque": [risque(p) if n == "Disponible" else "" for p, n in zip(vue.points, vue.niveau)]})
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        "Traitement": st.column_config.TextColumn(width=280),
        dispo_nom: st.column_config.NumberColumn(format="%.1f %%", width=108),
        prevu_nom: st.column_config.NumberColumn(format="%.1f %%", width=108),
        "Boîtes/mois": st.column_config.NumberColumn(format="%d", width=100),
        "Risque": st.column_config.TextColumn(width=80)})
    st.download_button("Télécharger cette liste", table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       "availscore_liste.csv", "text/csv")


def page_pipeline():
    chemin = f"{TRANSFORMEES}/journal_pipeline.csv"
    if not os.path.exists(chemin):
        st.info("Lance un calcul pour voir les étapes.")
        return
    st.write("Voici ce que la pipeline fait aux données, étape par étape.")
    st.dataframe(pd.read_csv(chemin).drop(columns="Pourquoi"), hide_index=True, width="stretch")
    st.markdown("#### Résultat final")
    for nom, titre in (("disponibilite_traitements.csv", "Télécharger le résultat par traitement"),
                       ("disponibilite_boites.csv", "Télécharger le résultat par boîte")):
        with open(f"{RESULTATS}/{nom}", "rb") as f:
            st.download_button(titre, f.read(), nom, "text/csv")


def ranger(fichier):
    """Range un fichier dans le bon dossier d'apres son nom."""
    nom = fichier.name.replace("Copie de ", "")
    bas = nom.lower()
    if bas.endswith(".xls"):
        dossier = "medicam"
    elif bas.endswith(".xlsx"):
        dossier = "ansm"
    elif nom.startswith("CIS") and bas.endswith(".txt"):
        dossier = "bdpm"
    elif "declarations" in bas:
        dossier = "ansm_data"
    else:
        return None
    with open(f"{DONNEES}/{dossier}/{nom}", "wb") as f:
        shutil.copyfileobj(fichier, f)
    return dossier


def manquants():
    bdpm = ["CIS_bdpm.txt", "CIS_CIP_bdpm.txt", "CIS_COMPO_bdpm.txt", "CIS_GENER_bdpm.txt", "CIS_MITM.txt", "CIS_CIP_Dispo_Spec.txt"]
    liste = [f for f in bdpm if not os.path.exists(f"{DONNEES}/bdpm/{f}")]
    for d, txt in (("medicam", "les fichiers Medic'AM (.xls)"), ("ansm", "les listes ANSM (.xlsx)")):
        if not [f for f in os.listdir(f"{DONNEES}/{d}") if not f.startswith(".")]:
            liste.append(txt)
    return liste


def page_donnees(infos):
    if infos:
        st.write(f"Données de {nom_mois(infos['premier_mois'])} à {nom_mois(infos['dernier_mois'])}, "
                 f"calculées le {infos['calcule_le']}.")
    fichiers = st.file_uploader("Dépose les nouveaux fichiers : Medic'AM (.xls), ANSM (.xlsx) ou BDPM (.txt)",
                                accept_multiple_files=True, type=["xls", "xlsx", "txt", "csv"])
    if fichiers and st.button("Enregistrer ces fichiers"):
        rejetes = [f.name for f in fichiers if ranger(f) is None]
        st.success(f"{len(fichiers) - len(rejetes)} fichier(s) enregistré(s).")
        if rejetes:
            st.warning("Fichiers non reconnus : " + ", ".join(rejetes))
    absents = manquants()
    if absents:
        st.info("Pour recalculer, il manque encore : " + ", ".join(absents) + ".")
    if st.button("Recalculer", type="primary", disabled=bool(absents)):
        with st.spinner("Calcul en cours, une à deux minutes..."):
            try:
                pipeline.calculer(DONNEES, RESULTATS, TRANSFORMEES)
            except Exception as e:
                st.error(f"Le calcul a échoué : {e}")
                return
        st.cache_data.clear()
        st.rerun()


st.title("AvailScore")
if not os.path.exists(f"{RESULTATS}/infos.json"):
    st.warning("Aucun résultat pour l'instant. Dépose les données brutes, puis lance le calcul.")
    page_donnees(None)
    st.stop()

T, M, S, infos = charger()
st.caption(f"Disponibilité des médicaments en ville, en France. Dernier mois connu : {nom_mois(infos['dernier_mois'])}.")
un, deux, quatre, trois = st.tabs(["Tester un médicament", "Tous les traitements", "Pipeline", "Mettre à jour les données"])
with un:
    page_medicament(T, M, S, infos)
with deux:
    page_liste(T, infos)
with quatre:
    page_pipeline()
with trois:
    page_donnees(infos)
st.divider()
st.caption("AvailScore ne dit pas si ta pharmacie a la boîte. Il dit si, en France, ce médicament continue d'être délivré "
           "normalement. Il ne remplace ni un pharmacien ni un médecin.")
