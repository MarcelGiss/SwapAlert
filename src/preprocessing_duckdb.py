import duckdb as ddb

# schema resultat:
# RESULTATX - ID der messung - EGAL
# ANALYTX - messungsart
# ERGEBNISF wert
# AUFTRAGX auftragsid
# ERFASSDAT timestamp
#
# schema auftrag
# AUFTRAGX - auftragsid
# PATISTAMMX - patientid


auftrag_path = "./data/Auftrag_01_24-07_25.csv"
resultat_path = "./data/Resultat_01_24-07_25.csv"

resultat = ddb.read_csv(resultat_path, delimiter=";")
auftrag = ddb.read_csv(auftrag_path, delimiter=";")

preprocessing_auftrag = """
WITH  as (SELECT
          ANALYTX as analyt,
          ERGEBNISF as messwert,
          AUFTRAGX as auftragsid,
          ERFASSDAT as messtimestamp
      FROM r
      WHERE ERGEBNISF IS NOT NULL)
      """
preprocessed_auftrag = resultat.query("r", preprocessing_auftrag)


preprocessed_auftrag.to_csv("./data/preprocessed_auftrag.csv")
