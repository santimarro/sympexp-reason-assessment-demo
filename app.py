from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, url_for
from spacy import displacy
from tqdm import tqdm

from src.api.api_calls import APIModule
from src.explanation.explanation import ExplanationModule
from src.ner.ner import NERModule
from src.prevalence.prevalence import PrevalenceModule

app = Flask(__name__)
app.config["SECRET_KEY"] = "topSecret"
app.secret_key = "topSecret"

api_module = APIModule()
ner_module = NERModule()
prevalence_module = PrevalenceModule()
explanation_module = ExplanationModule()

DEFAULT_LABEL_COLORS = {
    "Finding": "#7aecec",
    "Sign or Symptom": "#bfeeb7",
    "Age Group": "#feca74",
    "Population Group": "#ff9561",
    "Location": "#aa9cfc",
    "Temporal Concept": "#c887fb",
    "No Symptom Occurrence": "#9cc9cc",
}
options = {"ents": list(DEFAULT_LABEL_COLORS), "colors": DEFAULT_LABEL_COLORS}

label_map = {
    "Finding": "Finding",
    "Sign_or_Symptom": "Sign or Symptom",
    "Age_Group": "Age Group",
    "Population_Group": "Population Group",
    "Location": "Location",
    "Temporal_Concept": "Temporal Concept",
    "No_Symptom_Occurence": "No Symptom Occurrence",
}

info = {}
map_disease_code = {}
map_code_disease = {}


def get_symptoms(info):
    symptoms = []
    for e in info["catTermsMap"]:
        for s in e["terms"]:
            symptoms.append(s)
    return symptoms


def get_score(instance):
    return instance.score


# app = Flask(__name__)


@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")


# Endpoint for receiving all the inputs and initiating the pipeline
@app.route("/start_pipeline", methods=["POST"])
def start_pipeline():
    # Receiving inputs from frontend
    clinical_case = request.json.get("clinical_case")
    # explanation = request.json.get("explanation")
    diseases = request.json.get("diseases")
    disease_ids = request.json.get("disease_ids")

    for i in range(len(diseases)):
        map_disease_code[diseases[i]] = disease_ids[i]

    # do the inverse
    # map_code_disease = {v: k for k, v in map_disease_code.items()}

    correct_disease = diseases[0]

    diseases_info = {}
    symptoms_info = {}
    incorrect_symptoms = []
    incorrect_dict = {}

    for disease, disease_id in tqdm(zip(diseases, disease_ids)):
        if len(disease) < 1:
            continue
        disease_info = api_module.get_disease_info(disease_id)
        diseases_info[disease] = disease_info
        disease_symptoms = get_symptoms(disease_info)
        symptoms_info[disease] = disease_symptoms
        if disease == correct_disease:
            correct_symptoms = disease_symptoms
        else:
            incorrect_symptoms.extend(disease_symptoms)
            incorrect_dict[disease] = disease_symptoms

    print("Finished API calls to HPO")

    info["diseases"] = diseases
    info["correct_disease"] = correct_disease
    info["clinical_case"] = clinical_case
    info["correct_symptoms"] = correct_symptoms
    info["incorrect_symptoms"] = incorrect_symptoms
    info["incorrect_diseases"] = incorrect_dict

    return jsonify(
        {
            "message": "Pipeline started",
            "clinical_case": clinical_case,
            "step_one_url": url_for("step_one"),
        }
    )


# Endpoint for the first modular step of the AI pipeline (e.g., Named Entity Recognition)
@app.route("/step_one", methods=["POST"])
def step_one():
    # Recovering inputs from session
    print("Running NER")
    clinical_case = info["clinical_case"]
    named_entities = ner_module.get_sentences_nes_list(clinical_case)
    # Map the labels in named_entities to the labels in the frontend
    modified_named_entities = []
    for entity in named_entities:
        modified_named_entities.append(
            (label_map[entity[0]], entity[1], entity[2], entity[3])
        )
    named_entities = modified_named_entities
    # ('Sign_or_Symptom', 'slight malar hypertrichosis', 358, 385)
    # Filter named entities that are only Signs or Symptoms
    found_symptoms = [x[1] for x in named_entities if x[0] == "Sign or Symptom"]
    no_occurrence_symptoms = [
        x[1] for x in named_entities if x[0] == "No Symptom Occurence"
    ]

    info["present_symptoms"] = found_symptoms
    info["no_occurrence_symptoms"] = no_occurrence_symptoms

    incorrect_disease_names = ", ".join(list(info["incorrect_diseases"].keys()))
    correct_disease_name = info["correct_disease"]

    # Create a dictionary with the named entities in the Spacy NER format
    ent_input = {"text": clinical_case, "ents": [], "title": None}
    for entity in named_entities:
        ent_input["ents"].append(
            {"start": entity[2], "end": entity[3], "label": entity[0]}
        )

    html = displacy.render(
        ent_input, style="ent", page=True, manual=True, options=options
    )
    soup = BeautifulSoup(html, "lxml")
    html_body = soup.figure
    html_body.attrs = {}

    result_one = "NER performed"
    return jsonify(
        {
            "result": result_one,
            "clinical_case": clinical_case,
            "named_entities": named_entities,
            "correct_disease": correct_disease_name,
            "incorrect_diseases": incorrect_disease_names,
            "html_body": str(html_body),
            "step_two_url": url_for("step_two"),
        }
    )


# Endpoint for the second modular step of the AI pipeline (e.g., Another Processing Step)
@app.route("/step_two", methods=["POST"])
def step_two():
    print("Calculating prevalence")
    data = {
        "case": info["clinical_case"],
        "correct_symptoms": info["correct_symptoms"],
        "present_symptoms": info["present_symptoms"],
        "incorrect_symptoms": info["incorrect_symptoms"],
        "correct_disease": info["correct_disease"],
        "incorrect_diseases": info["incorrect_diseases"],
    }

    formal_symptoms = prevalence_module.calculate_prevalence(data)
    formal_symptoms.sort(key=get_score, reverse=True)
    list_symptoms = [vars(x) for x in formal_symptoms]

    info["formal_symptoms"] = formal_symptoms

    result_two = "Prevalence calculated"
    return jsonify(
        {
            "result": result_two,
            "step_three_url": url_for("generate_explanation"),
            "list_symptoms": list_symptoms,
        }
    )


# Endpoint for generating explanations
@app.route("/generate_explanation", methods=["POST"])
def generate_explanation():
    print("Generating explanation")
    formal_symptoms = info["formal_symptoms"]
    selected_symptoms = formal_symptoms[:5]

    explanations = explanation_module.generate_simpler_explanation(selected_symptoms)

    # replace disease ids with disease names in explanations
    for i, expl in enumerate(explanations):
        for disease_id in map_code_disease.keys():
            if disease_id in expl:
                explanations[i] = expl.replace(disease_id, map_code_disease[disease_id])

    # gpt_explanation = explanations
    gpt_explanation = explanation_module.generate_gpt_explanation(
        info["diseases"], info["correct_disease"], explanations
    )
    return jsonify({"explanations": gpt_explanation})


@app.route("/loading.html")
def loading():
    return render_template("loading.html")


@app.route("/results.html")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    app.run(debug=True)
