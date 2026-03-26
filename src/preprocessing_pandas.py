import pandas as pd

auftrag_path = "./data/Auftrag_01_24-07_25.csv"
resultat_path = "./data/Resultat_01_24-07_25.csv"

auftrag_df = pd.read_csv(auftrag_path, delimiter=";")
resultat_df = pd.read_csv(resultat_path, delimiter=";")

resultat.groupby("ANALYTX").sort("ANALYTX").size()
resultat["ANALYTX"].unique().size()