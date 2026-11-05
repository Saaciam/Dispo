# AvailScore : l'application

L'application calcule la disponibilité réelle d'un médicament : elle compare ce qui a été délivré à ce qui était attendu, puis prévoit les mois suivants.

## Lancer l'application

Ouvre un terminal dans ce dossier, puis tape :

```
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre dans ton navigateur. Les résultats d'août 2026 sont déjà inclus : tu peux tester un médicament tout de suite.

## Les quatre onglets

- **Tester un médicament** : tape un nom et choisis dans combien de mois tu veux la prévision.
- **Tous les traitements** : la liste des traitements disponibles, tendus ou en manque.
- **Pipeline** : les transformations appliquées, étape par étape, et le résultat final à télécharger.
- **Mettre à jour les données** : dépose un nouveau fichier, puis clique sur Recalculer.

## La pipeline

Pour la relancer sans l'interface, tape `python pipeline.py`. Elle prend une à deux minutes.

| Dossier | Contenu |
| --- | --- |
| `donnees/` | Les fichiers bruts : Medic'AM, ANSM, BDPM |
| `donnees_transformees/` | Le résultat de chaque étape |
| `resultats/` | Le résultat final |

Fichiers de `donnees_transformees/` :

1. `1_medicam_boites_par_mois.csv` : les boîtes par présentation et par mois, cases vides et négatifs corrigés.
2. `2_unites_par_specialite.csv` : les boîtes converties en unités, regroupées par spécialité.
3. `3_signalements_ansm_nettoyes.csv` : les signalements sans doublons, reliés à une spécialité.
4. `4_medicaments_et_traitements.csv` : chaque spécialité et son traitement.
5. `5_traitements_par_mois.csv` : une ligne par traitement et par mois, avec chaque terme de la formule.
6. `journal_pipeline.csv` : ce que chaque étape a changé.

Résultat final dans `resultats/` : `disponibilite_traitements.csv` et `disponibilite_boites.csv`, à ouvrir dans Excel. Les autres fichiers de ce dossier servent à l'application.

Open Medic n'est pas inclus : il pèse 1,4 Go et ne sert qu'à contrôler les totaux, pas à calculer la formule.

## Mettre à jour les données

Dépose chaque nouveau fichier Medic'AM dans l'onglet de mise à jour, puis clique sur Recalculer. Si un mois apparaît dans deux fichiers, le fichier le plus récent par son nom l'emporte.

Pour mettre l'application en ligne, envoie ce dossier sur GitHub, puis déclare `app.py` sur Streamlit Community Cloud.
