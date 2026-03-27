import duckdb as ddb
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
preprocessed_auftrag = "./data/basis2_marcel.csv"

preprocessed_auftrag = ddb.read_csv(preprocessed_auftrag, delimiter=",")
preview_df = preprocessed_auftrag.query("r", "SELECT * FROM r LIMIT 5").df()
print(preview_df)

# analyt  messwert  auftragsid  patientid       messtimestamp
#hier noch hinzufügen bei select varianz von messwert
sql = """
    Select * from r
    where patientid = 1707000
      """

sqlGetAuftrag = """
    Select * from r
    where auftragsid = 14220810
      """

## change person id 1707000 for auftragsid 14220810
sqlChangePerson = """
SELECT
  analyt,
  messwert,
  auftragsid,
  CASE
    WHEN auftragsid = 14220810 THEN 1707000
    ELSE patientid
  END AS patientid,
  messtimestamp
FROM r
"""

sql2 = """
    Select patientid, Count(*) as gesamt_anzahl, Count(distinct(analyt)) as unique_analyt, Count(distinct(auftragsid)) as unique_aufträge
      from r
      where mod(patientid, 3000) = 0
      and 2 < (Select Count(*) from r as r2 where r2.patientid = r.patientid and r2.analyt = r.analyt group by r2.patientid, r2.analyt)
      group by patientid
      having unique_aufträge > 2
      """

sqlCreateReduced = """
Select main.* from r main
join (
Select patientid, analyt from r
group by patientid, analyt
having Count(*) > 2
) as subq
on main.patientid = subq.patientid and main.analyt = subq.analyt
"""

sql_falsify = """WITH base AS (
  SELECT analyt, messwert, auftragsid, patientid, messtimestamp
  FROM r
),
patients AS (
  SELECT DISTINCT patientid
  FROM base
),
-- 1) Für jeden Patienten genau einen zufälligen *anderen* Patienten wählen
donor_per_patient AS (
  SELECT
    p.patientid AS target_patientid,
    d.patientid AS donor_patientid,
    ROW_NUMBER() OVER (
      PARTITION BY p.patientid
      ORDER BY RANDOM()
    ) AS rn
  FROM patients p
  JOIN patients d
    ON p.patientid <> d.patientid
),
chosen_donor AS (
  SELECT target_patientid, donor_patientid
  FROM donor_per_patient
  WHERE rn = 1
),
-- 2) Pro Spender-Patient eine zufällige auftragsid wählen
random_auftrag_per_donor AS (
  SELECT
    patientid AS donor_patientid,
    auftragsid,
    ROW_NUMBER() OVER (
      PARTITION BY patientid
      ORDER BY RANDOM()
    ) AS rn
  FROM (
    SELECT DISTINCT patientid, auftragsid
    FROM base
  )
),
chosen_auftrag AS (
  SELECT donor_patientid, auftragsid AS new_auftragsid
  FROM random_auftrag_per_donor
  WHERE rn = 1
)
SELECT
  b.analyt,
  b.messwert,
  ca.new_auftragsid AS auftragsid,   -- neue zufällige auftragsid
  b.patientid,
  b.messtimestamp,
  b.auftragsid AS original_auftragsid
FROM base b
JOIN chosen_donor cd
  ON b.patientid = cd.target_patientid
JOIN chosen_auftrag ca
  ON cd.donor_patientid = ca.donor_patientid;
"""

sql3 = """
WITH base AS (
  SELECT r.patientid, r.analyt, r.auftragsid, r.messwert, r.messtimestamp
  FROM r
  JOIN (
    SELECT patientid, analyt, COUNT(*) AS anzahl_messwerte
    FROM r
    GROUP BY patientid, analyt
    HAVING COUNT(*) > 2
  ) AS agg
    ON r.patientid = agg.patientid AND r.analyt = agg.analyt
),
stats AS (
  SELECT
    patientid,
    analyt,
    AVG(messwert) AS mean_messwert,
    STDDEV_POP(messwert) AS std_messwert
  FROM base
  GROUP BY patientid, analyt
  order by patientid
)
SELECT
  b.patientid,
  b.analyt,
  b.auftragsid,
  b.messwert,
  b.messtimestamp,
  s.mean_messwert AS mean,
  s.std_messwert AS stddev,
  ABS(b.messwert - s.mean_messwert) AS distanz_abs,
  CASE
    WHEN s.std_messwert IS NULL OR s.std_messwert = 0 THEN NULL
    ELSE ABS((b.messwert - s.mean_messwert) / s.std_messwert)
  END AS individualDistance
FROM base b
JOIN stats s
  ON b.patientid = s.patientid
 AND b.analyt = s.analyt
      """

## hinzufügen der Means pro PErson und Analyt zu jeder Zeile der Tabelle, dann die Standardabweichung
sql3_2 = """
Select * , Floor(individualDistance / 2) as ausreiserScore
from r
order by ausreiserScore desc
"""

sql4 = """ Select * from r
where patientid = 972440
ORDER BY analyt
      """

sql_mixed_false_data = """SELECT 
       floor(random() * 47169) + 1 AS patientid, analyt, messwert, auftragsid, messtimestamp
FROM r;
      """

sql_analyt_analyse = """
Select analyt, Count(*) as anzahl, mean(mean) as meanPerPerson, mean(distinct(stddev)) as stdPerPersonAvrg, stddev(distinct(stddev)) as deviationOfDeviations, mean(messwert) as totalMean, STDDEV_POP(messwert) as totalStddev, totalStddev/stdPerPersonAvrg as analytUsabilityscore, max(messwert) as maxMessw
from r
group by analyt
having totalStddev > 0
order by analytUsabilityscore desc

"""
sql_score_per_auftrag = """
Select auftragsid, patientid, mean(ausreiserScore), median(individualDistance), max(messtimestamp) from r 
group by auftragsid, patientid
order by mean(ausreiserScore) desc
"""

# Backward-compatible alias in case the old variable name is still used.
sql_analyt_anaylse = sql_analyt_analyse


analysis_results = preprocessed_auftrag.query("r", sql_score_per_auftrag).df()
print(analysis_results)
print(analysis_results.head())
#print(analysis_results.to_string(index=False))

analysis_results.to_csv("./data/basis2_end_marcel.csv", index=False)