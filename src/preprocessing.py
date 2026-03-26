import pandas as pd
import duckdb as ddb

auftrag_path = "./data/Auftrag_01_24-07_25.csv"
resultat_path = "./data/Resultat_01_24-07_25.csv"

auftrag_df = pd.read_csv(auftrag_path, delimiter=";")
resultat_df = pd.read_csv(resultat_path, delimiter=";")#,nrows=1000 )

resultat = ddb.read_csv(resultat_path, delimiter=";")

resultat.groupby("ANALYTX").sort("ANALYTX").size()

resultat["ANALYTX"].unique().size()


resultat = ddb.read_csv(resultat_path, delimiter=";")


import duckdb as ddb

resultat_path = "./data/Resultat_01_24-07_25.csv"

# Load the CSV → DuckDB relation
resultat = ddb.read_csv(resultat_path, delimiter=";")

sql = """
      SELECT
          ANALYTX,
          COUNT(*) AS cnt
      FROM r   
      GROUP BY ANALYTX
      ORDER BY cnt DESC;
      """

#  "r" is the name we expose inside the SQL string
summary_df = resultat.query("r", sql).df()
