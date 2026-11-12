# AvailScore : Power BI, puis Python, puis l'application

- donnees/            les fichiers bruts (Medic'AM, ANSM, BDPM)
- exports_power_bi/   les 7 CSV exportés de Power BI 
- python/             1_nettoyage_final.py, 2_machine_learning.py, moteur.py
- resultats/          les 3 fichiers propres, tableau_final.csv, notes_machine_learning.txt
- app.py              l'application Streamlit (modèle V4)

## Lancer l'application 
    python -m pip install -r requirements.txt     # une seule fois
    python -m streamlit run app.py

Le navigateur s'ouvre sur http://localhost:8501.