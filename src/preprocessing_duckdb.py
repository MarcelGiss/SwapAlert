import duckdb as ddb

auftrag_path = "./data/Auftrag_01_24-07_25.csv"
resultat_path = "./data/Resultat_01_24-07_25.csv"

resultat = ddb.read_csv(resultat_path, delimiter=";")
auftrag = ddb.read_csv(auftrag_path, delimiter=";")

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

# Export summary DataFrame to CSV
summary_path = "./data/summary_df.csv"
summary_df.to_csv(summary_path, index=False)
