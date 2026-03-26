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


# Paths to the CSV files containing the raw data
auftrag_path = "./data/Auftrag_01_24-07_25.csv"
resultat_path = "./data/Resultat_01_24-07_25.csv"

ddb.sql(f"CREATE OR REPLACE TABLE resultat AS SELECT * FROM read_csv_auto('{resultat_path}', delim=';')")
ddb.sql(f"CREATE OR REPLACE TABLE auftrag AS SELECT * FROM read_csv_auto('{auftrag_path}', delim=';')")

filtering_auftrag = """
WITH ergebnisse_mit_werten AS (
    SELECT
        ANALYTX as analyt,
        ERGEBNISF as messwert,
        AUFTRAGX as auftragsid,
        ERFASSDAT as messtimestamp,
        ROW_NUMBER() OVER (PARTITION BY auftragsid, analyt ORDER BY ERFASSDAT DESC) as rn
    FROM resultat
    WHERE ERGEBNISF IS NOT NULL
),
plausible_auftraege as (
    select auftragsid from (select auftragsid, max(rn) as maxrn from ergebnisse_mit_werten group by auftragsid) maxsubq
    where maxsubq.maxrn < 100
    group by maxsubq.auftragsid
)
SELECT analyt,
       messwert,
       ergebnisse_mit_werten.auftragsid,
       auftrag_distinct.PATISTAMMX as patientid,
       messtimestamp
FROM ergebnisse_mit_werten
inner join plausible_auftraege on
    plausible_auftraege.auftragsid = ergebnisse_mit_werten.auftragsid
inner join (select distinct AUFTRAGX, PATISTAMMX from auftrag) auftrag_distinct on
    auftrag_distinct.AUFTRAGX = ergebnisse_mit_werten.auftragsid
WHERE rn = 1
"""
ddb.sql(f"CREATE OR REPLACE VIEW resultat_filtered AS {filtering_auftrag}")

filtered_auftrag = ddb.sql(filtering_auftrag)

analyze_analyt_count_query = """
select analyt,
    count(auftragsid) as cnt
from resultat_filtered
group by analyt
"""
analyt_analyse = ddb.sql(analyze_analyt_count_query)
ddb.sql(f"CREATE OR REPLACE VIEW analyt_counts AS {analyze_analyt_count_query}")
analyt_analyse.to_csv("./data/analyt_analyse.csv")

cutoff_analyt = """
select * from resultat_filtered
    left join analyt_counts
    on analyt_counts.analyt = resultat_filtered.analyt
    where analyt_counts.cnt < 10000
"""
preprocessed_auftrag = ddb.sql(cutoff_analyt)

preprocessed_auftrag.to_csv("./data/preprocessed_auftrag.csv")

