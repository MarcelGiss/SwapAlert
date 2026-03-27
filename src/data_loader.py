import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, path: str, history_length: int = 20, n_rows: int | None = None):
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
        # Load the CSV, optionally limiting the number of rows for faster testing.
        if n_rows is not None:
            self.data = pd.read_csv(path, nrows=n_rows)
        else:
            self.data = pd.read_csv(path)
        self.all_analyte = self.data["analyt"].unique()
        self.history_length = history_length
        self.history_length = history_length

    def get_sample(self, add_synthetic_swap: bool = False):
        """selects a random patientid, and returns all the historical data for one patient
        returns a dataframe of size numberofanalyte x history_length
        where the analyte are columns, and all missing analyte per auftrag are NULL
        if there are less than history_length aufträge per this patientid, add aufträge with all null
        if add_synthetic_swap, one of the history_length aufträge, is swapped for a random other auftrag of any other patientid
        """

        # 1. Choose a random patient
        patient_ids = self.data["patientid"].unique()
        if len(patient_ids) == 0:
            raise ValueError("No patient data available")
        patient_id = np.random.choice(patient_ids)

        # 2. Filter data for this patient and sort chronologically
        patient_data = self.data[self.data["patientid"] == patient_id]
        # Ensure proper datetime handling for ordering
        if "messtimestamp" in patient_data.columns:
            patient_data = patient_data.copy()
            patient_data["messtimestamp"] = pd.to_datetime(patient_data["messtimestamp"], errors="coerce")
            patient_data = patient_data.sort_values("messtimestamp")
        else:
            patient_data = patient_data.sort_values("auftragsid")

        # 3. Determine the unique auftrags (orders) for this patient
        auftrags = patient_data["auftragsid"].unique()
        # Use the most recent `history_length` auftrags (if available)
        selected_auftrags = auftrags[-self.history_length:]

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

        # 7. If synthetic swap is requested, replace one random column with data from another patient
        if add_synthetic_swap:
            # Choose a column to replace (must be one of the real auftrags, not a pad)
            real_cols = [c for c in df_pivot.columns if not str(c).startswith("pad_")]
            if real_cols:
                swap_col = np.random.choice(real_cols)
                # Choose a random other patient
                other_patients = patient_ids[patient_ids != patient_id]
                if len(other_patients) > 0:
                    other_id = np.random.choice(other_patients)
                    other_data = self.data[self.data["patientid"] == other_id]
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
                        other_series = other_pivot.reindex(self.all_analyte).iloc[
                            :, 0] if not other_pivot.empty else pd.Series([np.nan] * len(self.all_analyte),
                                                                          index=self.all_analyte)
                        df_pivot[swap_col] = other_series.values

        # 8. Return a DataFrame with analytes as rows and exactly `history_length` columns
        # Ensure column order matches the original chronological order (if any) and truncate excess columns
        df_result = df_pivot.iloc[:, : self.history_length]
        return df_result


if __name__ == "__main__":
    # Simple test harness for the DataLoader class.
    # Load the sample CSV provided in the repository and retrieve a sample
    # without and with synthetic swap to verify basic functionality.
    csv_path = "./data/preprocessed_auftrag.csv"
    loader = DataLoader(csv_path, n_rows = 100)
    print("--- Sample without synthetic swap ---")
    sample = loader.get_sample()
    print(sample)
    print("Shape:", sample.shape)
    print("--- Sample with synthetic swap ---")
    sample_swapped = loader.get_sample(add_synthetic_swap=True)
    print(sample_swapped)
    print("Shape (swapped):", sample_swapped.shape)
