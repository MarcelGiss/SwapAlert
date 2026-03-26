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
WITH ergebnisse_mit_werten AS (
    SELECT
        ANALYTX as analyt,
        ERGEBNISF as messwert,
        AUFTRAGX as auftragsid,
        ERFASSDAT as messtimestamp,
        ROW_NUMBER() OVER (PARTITION BY auftragsid, analyt ORDER BY ERFASSDAT DESC) as rn
    FROM r
    WHERE ERGEBNISF IS NOT NULL
),
plausible_auftraege as (
    select auftragsid from (select auftragsid, max(rn) as maxrn from ergebnisse_mit_werten group by auftragsid) maxsubq
    where maxsubq.maxrn < 100
    group by maxsubq.auftragsid
)    
-- here, I want to not include and 
SELECT analyt, 
       messwert, 
       ergebnisse_mit_werten.auftragsid, 
       messtimestamp 
FROM ergebnisse_mit_werten
inner join plausible_auftraege on
    plausible_auftraege.auftragsid = ergebnisse_mit_werten.auftragsid
WHERE rn = 1
"""
preprocessed_auftrag = resultat.query("r", preprocessing_auftrag)
# preprocessed_auftrag.to_csv("./data/preprocessed_auftrag.csv")

print(preprocessed_auftrag.df())
