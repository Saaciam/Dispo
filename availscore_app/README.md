# AvailScore : Power BI, puis Python, puis l'application

Un pharmacien cherche un médicament (nom ou code CIP13) et choisit un mois (le mois en cours ou l'un des
5 suivants). L'application répond :
- **aujourd'hui** : rupture de stock, tension d'approvisionnement ou remise à disposition, d'après le statut
  publié par l'ANSM dans la BDPM (revérifié automatiquement toutes les 12 heures) ;
- **pour le mois choisi** : la probabilité que le laboratoire déclare à l'ANSM une rupture de stock ou un risque
  de rupture pendant ce mois, avec la part estimée de chacun (d'après le fichier Data.ansm « historique ruptures ») ;
- les **équivalents** et leur situation, pour choisir une substitution.

Les listes ANSM sont publiées une fois par an, avec beaucoup de retard sur les ventes Medic'AM. Chaque mois
proposé a donc son propre modèle, entraîné dans la même situation que celle du jour : déclarations vues avec le
même retard, même distance entre les dernières ventes et le mois prédit (moteur.situation).

- donnees/            les fichiers bruts : medicam/ (2021 à aujourd'hui), ansm/ (listes 2019 à 2025), ansm_historique/, bdpm/
- exports_power_bi/   les 7 tables (refaites automatiquement à partir de donnees/ par mise_a_jour.py)
- python/             1_nettoyage_final.py, 2_machine_learning.py
- resultats/          les fichiers propres, disponibilite_ansm.csv, manifeste.json (état des données traitées)
- moteur.py           variables, modèle V4, prédiction, performance des modèles
- mise_a_jour.py      traitement automatique : donnees/ -> exports_power_bi/ -> resultats/
- app.py              l'application Streamlit

## Lancer l'application (depuis le dossier availscore_app)
    cd availscore_app
    python -m pip install -r requirements.txt     # une seule fois
    python -m streamlit run app.py

Le navigateur s'ouvre sur http://localhost:8501.

## Mettre à jour les données
- **Automatique** : ajoutez ou remplacez un fichier dans donnees/bdpm, donnees/ansm ou donnees/medicam.
  Au lancement suivant, l'application le voit, vérifie le format, refait tout le traitement et réapprend le modèle.
- **Depuis l'application**, page « Mise à jour des données » : télécharger la dernière BDPM sur le site
  officiel, déposer de nouveaux fichiers, ou relancer le traitement.
- **En ligne de commande** : `python mise_a_jour.py` (ajouter `--telecharger` pour récupérer la dernière BDPM, les nouvelles listes ANSM et les nouveaux semestres Medic'AM).
