import pandas as pd
class DataLoader:

    def __init__(self, path:str ):
        self.data = pd.read_csv(path)
        self.all_analyte = self.data["analyt"].unique()

    def get_patient_history(self, add_synthetic_swap:bool = False):
        """selects a random patientid, and returns all the historical data for one patient
        if
        """
        pass

if __name__ == "__main__":
    loader = DataLoader("./data/preprocessed_auftrag.csv")