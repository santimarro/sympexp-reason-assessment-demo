import requests
import pandas as pd
from fuzzywuzzy import process

from src.config.config import Configuration

class APIModule:
    def __init__(self):
        self.config = Configuration()
        self.session = requests.Session()

    def search_disease(self, disease_name):
        url = "https://hpo.jax.org/api/hpo/search"
        params = {"q": disease_name, "category": "diseases"}
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            search_results = response.json()
            return search_results["diseases"]
        else:
            print(f"Error searching diseases for {disease_name}: {response.status_code}")
            return []

    def select_closest_disease(self, disease_name, search_results):
        disease_names = [entry["dbName"] for entry in search_results]

        closest_match = process.extractOne(disease_name, disease_names)
        if closest_match is not None:
            closest_name, _ = closest_match
            closest_names = process.extract(disease_name, disease_names, limit=2)

            find_orpha = False
            max_score = 0
            first_name, first_score = closest_names[0]

            if len(closest_names) == 2 and first_name == closest_names[1][0]:
                find_orpha = True
            
            for entry in search_results:
                if entry["dbName"] == closest_name and (not find_orpha or entry["db"] == "ORPHA"):
                    return entry["diseaseId"]

        return None

    def get_disease_info(self, disease_id):
        url = f"https://hpo.jax.org/api/hpo/disease/{disease_id}"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching disease info for {disease_id}: {response.status_code}")
            return None

    def get_disease_symptoms_by_id(self, disease_id):
        url = f"https://hpo.jax.org/api/hpo/disease/{disease_id}/phenotypes"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return [entry["phenotype"] for entry in response.json()]
        else:
            print(f"Error fetching symptoms for {disease_id}: {response.status_code}")
            return []

    def get_disease_data(self, disease_name):
        disease_data = []
        search_results = self.search_disease(disease_name)
        closest_disease_id = self.select_closest_disease(disease_name, search_results)
        if closest_disease_id is not None:
            disease_info = self.get_disease_info(closest_disease_id)
            if disease_info is not None:
                disease_data = disease_info

        return disease_data

    def get_disease_symptoms(self, disease_name):
        symptoms_list = []
        disease_data = self.get_disease_data(disease_name)
        if disease_data:
            for item in disease_data["catTermsMap"]:
                terms = item.get('terms', [])
                for term in terms:
                    symptom = term.get('name')
                    if symptom:
                        symptoms_list.append(symptom)

        return symptoms_list


    def make_api_calls(self):
        symptoms_df = pd.read_csv(self.config.SYMPTOMS_FILE_DIR)
        symptoms_df = symptoms_df[symptoms_df['diagnosis_question'] == 1].reset_index(drop=True)
        
        correct_disease = symptoms_df.apply(lambda row: row[f'option_{row["correct_answer"]}'], axis=1).str.replace('.', '')
        incorrect_diseases = symptoms_df.apply(lambda row: [str(row[f'option_{answer}']) for answer in range(1, 6) if answer != row['correct_answer']], axis=1).apply(lambda lst: [string.replace('.', '') for string in lst])

        A = correct_disease.apply(self.get_disease_symptoms)
        B = incorrect_diseases.apply(lambda row: [symptom for disease in row for symptom in self.get_disease_symptoms(disease)])
        C = symptoms_df['case_nes']

        symptoms_df['A'] = A
        symptoms_df['B'] = B
        symptoms_df['C'] = C

        D = symptoms_df.apply(lambda row: list(set(tuple(row['A'])).union(set(tuple(row['B']))).difference(set(row['C']))), axis=1)
        symptoms_df['D'] = D

        symptoms_df.to_csv(self.config.SETS_FILE_DIR)