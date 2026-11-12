# =============================================================================
# AvailScore : l'application
# Quel médicament risque d'être signalé en rupture (ou risque de rupture)
# à l'ANSM dans les 3 prochains mois ? Modèle utilisé : V4 (le meilleur au test).
#
# Lancer :  streamlit run app.py      (depuis le dossier AvailScore_PowerBI)
# =============================================================================

import os
import subprocess
import sys

import altair as alt
import pandas as pd
import streamlit as st

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "python"))
import moteur  # noqa: E402

RESULTATS = os.path.join(ICI, "resultats")
EXPORTS = os.path.join(ICI, "exports_power_bi")
EXPORTS_ATTENDUS = ["Medicam", "Signalements", "Presentations", "Specialites", "Compositions", "Generiques", "MITM"]

st.set_page_config(page_title="AvailScore", page_icon="💊", layout="wide")


# ---------- calculs (gardés en mémoire tant que les fichiers ne changent pas) ----------
def date_fichiers():
    return max(os.path.getmtime(os.path.join(RESULTATS, f)) for f in
               ["ventes_par_medicament.csv", "signalements_propres.csv", "fiches_medicaments.csv"])


@st.cache_data(show_spinner="Construction du tableau médicament × mois…")
def tableau(_date):
    ventes, signal, fiches = moteur.charger(RESULTATS)
    data, dernier = moteur.construire(ventes, signal, fiches)
    return data, dernier, fiches, ventes, signal


@st.cache_resource(show_spinner="Entraînement du modèle V4…")
def modele(_date):
    data = tableau(_date)[0]
    return moteur.entrainer(data)


@st.cache_data(show_spinner="Comparaison des 4 versions…")
def notes(_date):
    return moteur.evaluer(tableau(_date)[0])


@st.cache_data
def predictions(_date):
    data, _, fiches, _, _ = tableau(_date)
    mod, base = modele(_date)
    p = data[["cis", "mois", "historique_complet", "cible"] + moteur.VARIABLES].copy()
    p["risque"] = mod.predict_proba(p[moteur.VARIABLES])[:, 1]
    p["fois_moyenne"] = p.risque / base
    p["niveau"] = pd.cut(p.fois_moyenne, [-1, 1.5, 3, 1e9], labels=["Faible", "À surveiller", "Élevé"]).astype(str)
    return p.merge(fiches[["cis", "nom", "titulaire", "atc", "traitement"]], on="cis")


if not os.path.exists(os.path.join(RESULTATS, "fiches_medicaments.csv")):
    st.error("Les fichiers propres manquent dans resultats/. Va dans l'onglet « Mettre à jour » "
             "ou lance `python python/1_nettoyage_final.py`.")
    st.stop()

DATE = date_fichiers()
data, dernier_signal, fiches, ventes, signal = tableau(DATE)
mod, base = modele(DATE)
P = predictions(DATE)
mois_dispo = sorted(P.mois.unique())


def pct(x):
    """0.094 -> '9,4 %' (écriture française)"""
    return f"{x * 100:.1f} %".replace(".", ",")


def fin_horizon(m):
    p = pd.Period(m, "M")
    return f"{(p + 1).strftime('%m/%Y')} à {(p + 3).strftime('%m/%Y')}"


# ---------- barre latérale ----------
with st.sidebar:
    st.title("💊 AvailScore")
    st.caption("Risque qu'un médicament soit signalé à l'ANSM (rupture ou risque de rupture) "
               "dans les 3 mois qui suivent.")
    defaut = mois_dispo.index(dernier_signal) if dernier_signal in mois_dispo else len(mois_dispo) - 1
    mois = st.selectbox("Situation à la fin du mois de", mois_dispo, index=defaut,
                        format_func=lambda m: pd.Period(m, "M").strftime("%m/%Y"))
    st.write(f"**Prédiction pour : {fin_horizon(mois)}**")
    if mois > dernier_signal:
        st.warning(f"Les listes ANSM s'arrêtent en {pd.Period(dernier_signal, 'M').strftime('%m/%Y')}. "
                   "Après cette date, le modèle ne voit pas les signalements récents : "
                   "il sous-estime le risque des médicaments déjà signalés.")
    st.divider()
    st.caption(f"Modèle : V4 (machine learning). En moyenne, {str(round(base * 100, 1)).replace('.', ',')} % des médicaments sont signalés "
               "dans les 3 mois. « Élevé » = au moins 3 fois cette moyenne ; "
               "« À surveiller » = au moins 1,5 fois.")

M = P[P.mois == mois].sort_values("risque", ascending=False).reset_index(drop=True)
M.index += 1

onglets = st.tabs(["📋 Liste à surveiller", "🔎 Un médicament", "📊 Le modèle", "🔄 Mettre à jour"])

# ---------- 1. la liste ----------
with onglets[0]:
    st.subheader(f"Médicaments les plus à risque pour {fin_horizon(mois)}")
    c = st.columns(4)
    c[0].metric("Médicaments suivis", f"{len(M):,}".replace(",", " "))
    c[1].metric("Risque élevé", int((M.niveau == "Élevé").sum()))
    c[2].metric("À surveiller", int((M.niveau == "À surveiller").sum()))
    c[3].metric("Risque moyen", pct(M.risque.mean()))

    f1, f2, f3 = st.columns([2, 1, 1])
    texte = f1.text_input("Chercher un nom, un laboratoire ou un code ATC", "")
    niveaux = f2.multiselect("Niveau", ["Élevé", "À surveiller", "Faible"], default=["Élevé", "À surveiller"])
    seulement_mitm = f3.checkbox("Seulement les MITM", False)

    V = M[M.niveau.isin(niveaux)]
    if seulement_mitm:
        V = V[V.mitm == 1]
    if texte:
        t = texte.lower()
        V = V[V.nom.str.lower().str.contains(t, regex=False) | V.titulaire.str.lower().str.contains(t, regex=False)
              | V.atc.fillna("").str.lower().str.startswith(t)]

    vue = pd.DataFrame({
        "Médicament": V.nom, "Risque": (V.risque * 100).round(1), "Niveau": V.niveau,
        "Pourquoi": V.apply(moteur.raisons, axis=1), "× moyenne": V.fois_moyenne.round(1),
        "Laboratoire": V.titulaire, "ATC": V.atc})
    st.dataframe(vue, width="stretch", height=520,
                 column_config={"Risque": st.column_config.ProgressColumn("Risque (%)", min_value=0, max_value=100,
                                                                          format="%.1f %%")})
    st.caption(f"{len(vue)} médicaments affichés. Au test, parmi les 100 premiers de la liste, "
               "environ 1 sur 3 a vraiment été signalé dans les 3 mois. C'est 4 fois mieux que le hasard, "
               "mais pas une certitude : c'est une liste à surveiller.")
    st.download_button("Télécharger la liste (CSV)", vue.to_csv(index_label="Rang").encode("utf-8-sig"),
                       f"availscore_risque_{mois}.csv", "text/csv")

# ---------- 2. un médicament ----------
with onglets[1]:
    noms = M.sort_values("nom")
    choix = st.selectbox("Choisis un médicament (tape quelques lettres)", noms.cis,
                         format_func=lambda c: f"{noms.set_index('cis').nom[c]} ({c})")
    r = M[M.cis == choix].iloc[0]
    rang = int(M.index[M.cis == choix][0])

    c = st.columns(4)
    c[0].metric("Risque sur 3 mois", pct(r.risque))
    c[1].metric("Par rapport à la moyenne", f"× {r.fois_moyenne:.1f}".replace(".", ","))
    c[2].metric("Niveau", r.niveau)
    c[3].metric("Rang", f"{rang} / {len(M)}")
    st.write(f"**{r.nom}** — {r.titulaire} — ATC {r.atc}")
    st.info("Signes présents : " + moteur.raisons(r) + ".  \nLe modèle les pèse avec les autres informations : "
            "un signe isolé ne suffit pas à rendre le risque élevé.")

    with st.expander("Le détail des informations vues par le modèle"):
        st.table(pd.DataFrame({
            "Information": ["Ventes du mois / mois normal", "Ventes des 3 derniers mois / normal",
                            "Ventes instables (0 = très régulières)", "Mois avec signalement (3 derniers mois)",
                            "Mois avec signalement (12 derniers mois)", "Mois avec signalement (24 derniers mois)",
                            "Part de la famille ATC signalée ce trimestre", "MITM", "Un seul laboratoire",
                            "Injectable"],
            "Valeur": [f"{r.ventes_mois:.0%}", f"{r.ventes_3mois:.0%}", f"{r.ventes_instables:.2f}",
                       int(r.signal_3mois), int(r.signal_12mois), int(r.signal_24mois),
                       f"{r.classe_signalee:.0%}", "oui" if r.mitm else "non",
                       "oui" if r.un_seul_labo else "non", "oui" if r.injectable else "non"]}).set_index("Information"))

    g1, g2 = st.columns(2)
    hv = ventes[ventes.cis == choix]
    hv = hv.assign(date=pd.to_datetime(hv.mois + "-01"))
    sg = signal[signal.cis == choix].assign(date=lambda x: x.date.dt.to_period("M").dt.to_timestamp())
    sg = sg.drop_duplicates("date").merge(hv[["date", "boites"]], on="date", how="inner")
    ligne = alt.Chart(hv).mark_line().encode(x=alt.X("date:T", title=None), y=alt.Y("boites:Q", title="Boîtes remboursées"))
    points = alt.Chart(sg).mark_point(color="red", size=90, filled=True).encode(x="date:T", y="boites:Q",
                                                                              tooltip=[alt.Tooltip("date:T", format="%m/%Y")])
    g1.write("**Boîtes remboursées par mois** (points rouges = mois avec un signalement ANSM)")
    g1.altair_chart(ligne + points, width="stretch")

    hr = P[P.cis == choix].assign(date=lambda x: pd.to_datetime(x.mois + "-01"))
    g2.write("**Risque prédit, mois après mois**")
    courbe = alt.Chart(hr).mark_line(color="#d95f02").encode(
        x=alt.X("date:T", title=None), y=alt.Y("risque:Q", title="Risque sur 3 mois", axis=alt.Axis(format="%")))
    limite = alt.Chart(pd.DataFrame({"date": [pd.to_datetime(dernier_signal + "-01")]})).mark_rule(
        color="gray", strokeDash=[4, 4]).encode(x="date:T")
    g2.altair_chart(courbe + limite, width="stretch")
    g2.caption("Pointillé : fin des listes ANSM. Après, le risque est sous-estimé pour les médicaments déjà signalés.")

    meme = M[(M.traitement == r.traitement) & (M.cis != choix)]
    if len(meme):
        st.write("**Les équivalents (même traitement) et leur risque**")
        st.dataframe(pd.DataFrame({"Médicament": meme.nom, "Laboratoire": meme.titulaire,
                                   "Risque (%)": (meme.risque * 100).round(1), "Niveau": meme.niveau}),
                     width="stretch", hide_index=True)

# ---------- 3. le modèle ----------
with onglets[2]:
    st.subheader("Pourquoi la version 4 ?")
    tab_notes, n_test, part = notes(DATE)
    st.write(f"On a comparé 4 versions sur des mois **jamais vus pendant l'apprentissage** "
             f"(août et septembre 2025, {n_test:_} lignes, dont {pct(part)} signalées).".replace("_", " "))
    st.dataframe(tab_notes.style.format({"AUC": "{:.3f}", "AP": "{:.3f}", "Top 100": "{:.0%}"})
                 .highlight_max(color="#c7e9c0"), width="stretch")
    st.markdown("""
- **AUC** : on prend au hasard un médicament signalé et un non signalé. C'est la chance que le modèle
  donne le plus gros risque au bon. 0,5 = pile ou face, 1 = parfait.
- **AP** : la qualité du haut de la liste, de 0 à 1. Le hasard vaut la part de signalés (environ 0,09).
- **Top 100** : parmi les 100 médicaments les plus à risque, la part vraiment signalée dans les 3 mois.

La **V4 gagne sur les trois notes**, et elle gagnait aussi lors de la validation croisée dans le temps.
C'est elle que l'application utilise. Pour la mise en service, elle réapprend sur tous les mois dont on connaît
la réponse (juillet 2024 à septembre 2025).

**Les limites, à dire à l'oral.** Les listes ANSM s'arrêtent fin 2025. L'historique est court (15 mois avec
réponse). Le modèle repère surtout les médicaments MITM, déjà signalés, ou dont les ventes baissent. Une
rupture totalement inattendue reste difficile à prévoir.
""")

# ---------- 4. mettre à jour ----------
with onglets[3]:
    st.subheader("Mettre à jour avec de nouveaux exports Power BI")
    st.write("1. Dans Power BI, actualise les requêtes, puis exporte les 7 tables avec DAX Studio.  \n"
             "2. Dépose les 7 CSV ici.  \n"
             "3. Clique sur **Relancer le nettoyage**. L'application réapprend le modèle toute seule.")
    fichiers = st.file_uploader("Les 7 exports (Medicam.csv, Signalements.csv, Presentations.csv, "
                                "Specialites.csv, Compositions.csv, Generiques.csv, MITM.csv)",
                                type="csv", accept_multiple_files=True)
    if fichiers:
        for f in fichiers:
            nom = os.path.splitext(f.name)[0]
            if nom in EXPORTS_ATTENDUS:
                with open(os.path.join(EXPORTS, f.name), "wb") as out:
                    out.write(f.getbuffer())
                st.write(f"✅ {f.name} enregistré")
            else:
                st.write(f"⚠️ {f.name} ignoré : nom inattendu")
    presents = [n for n in EXPORTS_ATTENDUS if os.path.exists(os.path.join(EXPORTS, n + ".csv"))]
    st.caption(f"Exports présents : {len(presents)}/7")
    if st.button("Relancer le nettoyage", type="primary", disabled=len(presents) < 7):
        with st.spinner("Nettoyage Python en cours…"):
            res = subprocess.run([sys.executable, "1_nettoyage_final.py"], cwd=os.path.join(ICI, "python"),
                                 capture_output=True, text=True)
        if res.returncode == 0:
            st.success("Terminé.")
            st.code(res.stdout)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.button("Recharger l'application")
        else:
            st.error("Le nettoyage a échoué :")
            st.code(res.stderr[-3000:])
