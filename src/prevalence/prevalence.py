import pickle

from flair.embeddings import SentenceTransformerDocumentEmbeddings
from tqdm import tqdm

from src.config.config import Configuration
from src.matching_module.matching_module import match_with_context


class Symptom:
    def __init__(
        self,
        name="",
        disease="",
        correct=True,
        occurrence_rate=50,
        occurrence_rate_str="Frequent",
        definition="",
    ):
        self.name = name
        self.disease = disease
        self.correct = correct
        self.occurrence_rate = occurrence_rate
        self.occurrence_rate_str = occurrence_rate_str
        self.score = 0
        self.high_occurrence_rate = False
        self.low_occurrence_rate = False
        self.unique_to_correct = False
        self.definition = definition
        self.present_in_case = False

    def __str__(self):
        return self.name

    def show_info(self):
        print("Name:", self.name)
        print("Disease:", self.disease)
        print("Correct:", self.correct)
        print("Occurrence Rate:", self.occurrence_rate)
        print("Score:", self.score)
        print("High Occurrence Rate:", self.high_occurrence_rate)
        print("Low Occurrence Rate:", self.low_occurrence_rate)
        print("Unique to Correct:", self.unique_to_correct)
        print("Definition:", self.definition)
        print("Present in Case:", self.present_in_case)


class PrevalenceModule:
    def __init__(self):
        self.config = Configuration()
        self.high_occurrence_rate = 80
        self.low_occurrence_rate = 10
        self.occurrence_rate_dict = {
            "ORPHA": {
                "always present": 100,
                "very frequent": 90,
                "frequent": 55,
                "occasional": 15,
                "rare": 4,
                "very rare": 1,
                "excluded": 0,
            },
        }
        self.embedding_model = SentenceTransformerDocumentEmbeddings(
            self.config.embeddings_model
        )

        with open(self.config.HPO_EMBEDDINGS, "rb") as f:
            self.hpo_terms = pickle.load(f)
            f.close()

    def find_symptoms(self, formal_symptoms, case, present_symptoms):
        # Check if its present
        # Use matching module from ICAART to match the symptoms.
        case = {"case": case, "detected_symptoms": present_symptoms}
        bests = match_with_context(case, self.embedding_model, self.filtered_hpo_terms)
        for result in bests:
            for formal_symptom in formal_symptoms:
                if formal_symptom.name == result[0]:
                    formal_symptom.present_in_case = True
                    break

    def find_symptom(self, formal_symptom, case, present_symptoms):
        # Check if its present
        # Use matching module from ICAART to match the symptoms.
        case = {"case": case, "detected_symptoms": present_symptoms}
        bests = match_with_context(case, self.embedding_model, self.filtered_hpo_terms)
        for result in bests:
            if formal_symptom.name == result[0]:
                formal_symptom.present_in_case = True  # False
                break

    def create_formal_symptoms(self, diseases_info, correct=True):
        list_formal_symptoms = []
        for disease_name, symptoms_list in diseases_info.items():
            for symptom in symptoms_list:
                occurrence_rate = 50
                occurrence_rate_str = "Frequent"

                if "ORPHA" in symptom["sources"]:
                    occurrence_rate = self.occurrence_rate_dict["ORPHA"][
                        symptom["frequency"].lower()
                    ]
                    occurrence_rate_str = symptom["frequency"]

                formal_symptom = Symptom(
                    name=symptom["name"],
                    disease=disease_name,
                    correct=correct,
                    occurrence_rate=occurrence_rate,
                    occurrence_rate_str=occurrence_rate_str,
                    definition=symptom["definition"],
                )
                list_formal_symptoms.append(formal_symptom)

        return list_formal_symptoms

    def check_unique_to_correct_symptom(self, correct_symptoms, incorrect_symptoms):
        unique = True
        for correct_symptom in correct_symptoms:
            for incorrect_symptom in incorrect_symptoms:
                if correct_symptom.name == incorrect_symptom.name:
                    break
            correct_symptom.unique_to_correct = unique

    def calculate_prevalence(self, data):
        # key_causes = data['causes']
        case = data["case"]

        correct_symptoms = data["correct_symptoms"]
        present_symptoms = data["present_symptoms"]

        incorrect_symptoms = data["incorrect_symptoms"]

        all_symptoms = correct_symptoms + incorrect_symptoms
        all_symptoms_id = [s["ontologyId"] for s in all_symptoms]

        self.filtered_hpo_terms = {
            key: value
            for key, value in self.hpo_terms.items()
            if key in all_symptoms_id
        }

        correct_symptoms = {data["correct_disease"]: data["correct_symptoms"]}
        incorrect_symptoms = data["incorrect_diseases"]
        correct_formal_symptoms = self.create_formal_symptoms(
            correct_symptoms, correct=True
        )
        incorrect_formal_symptoms = self.create_formal_symptoms(
            incorrect_symptoms, correct=False
        )

        self.check_unique_to_correct_symptom(
            correct_formal_symptoms, incorrect_formal_symptoms
        )
        list_formal_symptoms = []

        self.find_symptoms(correct_formal_symptoms, case, present_symptoms)
        self.find_symptoms(incorrect_formal_symptoms, case, present_symptoms)

        for formal_symptom in tqdm(correct_formal_symptoms):
            # Check if kp is in the correct list of symptoms
            if formal_symptom.present_in_case:
                formal_symptom.score += 2

                if formal_symptom.occurrence_rate > self.high_occurrence_rate:
                    formal_symptom.score += 1
                    formal_symptom.high_occurrence_rate = True

                if formal_symptom.unique_to_correct:
                    formal_symptom.score += 1

            list_formal_symptoms.append(formal_symptom)

        for formal_symptom in tqdm(incorrect_formal_symptoms):
            if not formal_symptom.present_in_case:
                if formal_symptom.occurrence_rate > self.high_occurrence_rate:
                    formal_symptom.high_occurrence_rate = True
                    formal_symptom.score += 1
                else:
                    formal_symptom.score -= 1
            # Present in the case but its a symptom of an incorrect disease
            else:
                formal_symptom.score -= 1

            list_formal_symptoms.append(formal_symptom)
            # if formal_symptom.occurrence_rate < self.low_occurrence_rate:
            #     formal_symptom.score += 1
            #     formal_symptom.low_occurrence_rate = True

            # Find symptoms that are not in the clinical case but are in the incorrect symptoms.
            # For each one of those see if their occurrence rate is high, meaning that the fact that it is not
            # present makes a case for discarding that disease.
        return list_formal_symptoms


"""

1. IF the kp is in the correct list of symptoms: +2
ELSE the kp is not in the correct list of symptoms: -1
ENDIF
2. IF the kp has a high occurrence rate (more than XX%): +1
. ELSE the kp has a low occurrence rate (less than YY%): -1
ENDIF
3. IF the kp is in the correct list of symptoms AND the kp is not in the list of symptoms for incorrect diseases: +1        
4. IF the kp is in the incorrect list of symtptoms AND the kp is not in the present symptoms: +1


Algorithm:
For each symptom found in HPO, correct and incorrect ones, generate a Symptom instance and compute the score:



https://www.orpha.net/consor/cgi-bin/Disease_HPOTerms.php?lng=EN
The frequency in the patients' population can be :
- always present: 100%
- very frequent: 99%-80%
- frequent: 79%-30%
- occasional: 29%-5%
- rare: 4%-1%
"""
