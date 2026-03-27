import pandas as pd
import numpy as np

# Configure pandas to display the full DataFrame when printed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class DataLoader:
    def __init__(self, path: str, history_length: int = 20, n_rows: int | None = None, required_auftraege_per_patient=5):
        """Create a DataLoader.

        Parameters
        ----------
        path: str
            Path to the CSV file containing the raw data.
        history_length: int, default 20
            Number of historical orders (auftrags) to include in each sample.
        n_rows: int | None, optional
            If provided, only the first ``n_rows`` rows of the CSV are read.
            This can be useful for quick testing on a subset of the data.
        """
        # Load the CSV (optionally limiting rows for quick tests).
        if n_rows is not None:
            raw_df = pd.read_csv(path, nrows=n_rows)
            print(f"[DataLoader] Loaded CSV '{path}' (first {n_rows} rows) – {len(raw_df)} rows total.")
        else:
            raw_df = pd.read_csv(path)
            print(f"[DataLoader] Loaded CSV '{path}' – {len(raw_df)} rows total.")

        # Filter out patients with too few distinct orders.
        if "patientid" in raw_df.columns and "auftragsid" in raw_df.columns:
            patient_auftrag_counts = raw_df.groupby("patientid")["auftragsid"].nunique()
            valid_patients = patient_auftrag_counts[patient_auftrag_counts >= required_auftraege_per_patient].index
            raw_df = raw_df[raw_df["patientid"].isin(valid_patients)]
        # Suppressed verbose output for filtered patient summary.

        # Store a list of patient IDs for fast random selection.
        self._patient_ids = raw_df["patientid"].unique()
        # Suppressed verbose output for patient preparation summary.

        # Pre‑sort each patient's data once to avoid repeated sorting in ``get_sample``.
        # The original implementation filtered and sorted each patient individually,
        # which is O(N_patients) passes over the DataFrame and can be very slow for large
        # datasets. Instead we globally sort the DataFrame once and then split it into
        # patient groups, preserving the sort order. This reduces the work to a single
        # sort operation and a single group‑by pass.
        #
        # We also add an optional progress bar (tqdm) so the user sees activity while
        # the preprocessing runs. If tqdm is not installed we fall back to a no‑op
        # iterator.
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            # Define a minimal placeholder that mimics tqdm's interface.
            def tqdm(iterable, **kwargs):  # type: ignore
                return iterable

        # Determine sorting keys: timestamp if available, otherwise order_id.
        if "messtimestamp" in raw_df.columns:
            raw_df["messtimestamp"] = pd.to_datetime(
                raw_df["messtimestamp"], errors="coerce"
            )
            sort_keys = ["patientid", "messtimestamp"]
        else:
            sort_keys = ["patientid", "auftragsid"]

        # Global sort – this is much faster than per‑patient sorting.
        raw_df = raw_df.sort_values(sort_keys)

        # Build a dictionary of patient_id -> sorted patient DataFrame.
        self._patient_data: dict = {}
        # ``groupby`` preserves the order of rows within each group because we have
        # already sorted by the grouping key(s).
        for pid, patient_df in tqdm(
            raw_df.groupby("patientid"), desc="Pre‑sorting patient data"
        ):
            # ``patient_df`` is already sorted according to ``sort_keys``.
            self._patient_data[pid] = patient_df.copy()

        # Determine the set of all analytes, sorted by overall frequency.
        self.all_analyte = raw_df["analyt"].value_counts().index.to_numpy()
        # Suppressed verbose output for analyte count.
        self.history_length = history_length

    def get_sample(self, add_synthetic_swap: bool = False):
        """selects a random patientid, and returns all the historical data for one patient
        returns a dataframe of size numberofanalyte x history_length
        where the analyte are columns, and all missing analyte per auftrag are NULL
        if there are less than history_length aufträge per this patientid, add aufträge with all null
        if add_synthetic_swap, one of the history_length aufträge, is swapped for a random other auftrag of any other patientid
        """

        # 1. Choose a random patient from the pre‑filtered list.
        if len(self._patient_ids) == 0:
            raise ValueError("No patient data available")
        patient_id = np.random.choice(self._patient_ids)
        # Silent mode: omit per‑sample print statements to reduce console clutter.

        # 2. Retrieve the pre‑sorted data for this patient.
        patient_data = self._patient_data[patient_id]

        # 3. Determine the unique auftrags (orders) for this patient
        auftrags = patient_data["auftragsid"].unique()
        # Use the most recent `history_length` auftrags (if available)
        selected_auftrags = auftrags[-self.history_length:]
        # Silent mode: omit per‑sample details.

        # 4. Build a pivot table: rows = analyte, columns = auftrags, values = messwert
        # Filter to the selected auftrags first
        pivot_data = patient_data[patient_data["auftragsid"].isin(selected_auftrags)]
        df_pivot = pivot_data.pivot_table(
            index="analyt",
            columns="auftragsid",
            values="messwert",
            aggfunc="first",
        )

        # 5. Ensure all analytes are present as rows
        df_pivot = df_pivot.reindex(self.all_analyte)

        # 6. Pad missing columns if we have fewer than history_length
        current_cols = list(df_pivot.columns)
        missing = self.history_length - len(current_cols)
        if missing > 0:
            # create placeholder column names that do not clash with existing ones
            pad_names = [f"pad_{i}" for i in range(missing)]
            for name in pad_names:
                df_pivot[name] = np.nan
            # reorder columns to keep chronological order followed by pads
            df_pivot = df_pivot[current_cols + pad_names]
                # Optional: could log padding info if needed.
        if add_synthetic_swap:
            # Silent mode: omit detailed synthetic swap logs.
            # Choose a column to replace (must be one of the real auftrags, not a pad)
            real_cols = [c for c in df_pivot.columns if not str(c).startswith("pad_")]
            if real_cols:
                swap_col = np.random.choice(real_cols)
                # Choose a random other patient
                other_patients = self._patient_ids[self._patient_ids != patient_id]
                if len(other_patients) > 0:
                    other_id = np.random.choice(other_patients)
                    other_data = self._patient_data[other_id]
                    # Random auftrag from other patient
                    other_auftrags = other_data["auftragsid"].unique()
                    if len(other_auftrags) > 0:
                        other_auftrag = np.random.choice(other_auftrags)
                        other_subset = other_data[other_data["auftragsid"] == other_auftrag]
                        other_pivot = other_subset.pivot_table(
                            index="analyt",
                            columns="auftragsid",
                            values="messwert",
                            aggfunc="first",
                        )
                        # Align to full analyte list
                        other_series = (
                            other_pivot.reindex(self.all_analyte).iloc[:, 0]
                            if not other_pivot.empty
                            else pd.Series([np.nan] * len(self.all_analyte), index=self.all_analyte)
                        )
                        df_pivot[swap_col] = other_series.values
                        # Swap operation performed silently.

        # 8. Return a DataFrame with analytes as rows and exactly `history_length` columns
        # Ensure column order matches the original chronological order (if any) and truncate excess columns
        df_result = df_pivot.iloc[:, : self.history_length]
        return df_result


if __name__ == "__main__":
    # Simple test harness for the DataLoader class.
    # Load the sample CSV provided in the repository and retrieve a sample
    # without and with synthetic swap to verify basic functionality.
    csv_path = "./data/preprocessed_auftrag.csv"
    loader = DataLoader(csv_path, n_rows = 10000, required_auftraege_per_patient=3, history_length=3)
    # Demonstration: only report shapes to avoid overwhelming output.
    print("--- Sample without synthetic swap (shape only) ---")
    sample = loader.get_sample()
    print("Shape:", sample.shape)
    print("--- Sample with synthetic swap (shape only) ---")
    sample_swapped = loader.get_sample(add_synthetic_swap=True)
    print("Shape (swapped):", sample_swapped.shape)
