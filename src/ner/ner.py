from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import pipeline
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from src.config.config import Configuration
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from thefuzz import fuzz

# import nltk

import ast
import numpy as np
import os
import pandas as pd
import shutil
import torch
import warnings

from sentence_transformers import SentenceTransformer

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


class NERModule:
    """
    TO DO:
        1) Send ner_pipeline params to config
        2) Add config parameter to choose ner method
    """

    def __init__(self, config=None):
        self.config = config or Configuration()
        # self.transformer = transformer or self.config.TOKENIZING_MODEL
        # self.threshold = threshold or self.config.threshold
        # self.output_dir = os.path.join(
        #     self.config.EXPERIMENTS_DIR, self.transformer, str(self.threshold)
        # )
        # self.output_dir_dataframe = os.path.join(
        #     self.config.EXPERIMENTS_DIR, self.transformer)

        # self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        # self.model = AutoModel.from_pretrained(transformer)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # print(self.device)
        # self.model.to(self.device)

        # self.current_epoch = 1
        # self.sent_bert_model = SentenceTransformer(self.transformer, device=self.device)

        self.ner_pipeline = pipeline(
            "token-classification",
            model="smarro/medical_ner",
            use_auth_token=self.config.auth_token,
            device=self.device,
        )

    # def get_embeddings(self, text):
    #     tokens = self.tokenizer.tokenize(text)
    #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     input_tensor = torch.tensor([input_ids])

    #     with torch.no_grad():
    #         outputs = self.model(input_tensor)

    #     batch_size, seq_len, hidden_size = outputs.last_hidden_state.shape
    #     embeddings = outputs.last_hidden_state.reshape(
    #         batch_size * seq_len, hidden_size
    #     )

    #     return embeddings

    # def get_full_text_nes(self, data, column):
    #     nes = ner_prediction(corpus=data[column], compute=self.device)
    #     if nes.to_dict() != {}:
    #         # keep only sign_symptom entities
    #         if self.config.keep_only_symptoms:
    #             nes = nes[nes["entity_group"] == "Sign_symptom"]
    #         return nes["value"].tolist()
    #     else:
    #         return []
    # return nes

    # def split_sentences(self, clinical_case):
    #     # sentences = data[column].split(". ")
    #     sentences = nltk.sent_tokenize(clinical_case)
    #     return sentences

    # def compute_jaccard(self, case_sentence, explanation_sentence):
    #     case_sentence = set(case_sentence.split())
    #     explanation_sentence = set(explanation_sentence.split())
    #     return float(len(case_sentence.intersection(explanation_sentence)) / len(case_sentence.union(explanation_sentence)))

    # def compute_fuzzy_token_set_ratio(self, case_sentence, explanation_sentence):
    #     return fuzz.token_set_ratio(case_sentence, explanation_sentence)

    def detokenize_bert(self, sentence):
        pretok_sent = ""
        tokens = sentence.split()
        for tok in tokens:
            if tok.startswith("##"):
                pretok_sent += tok[2:]
            else:
                pretok_sent += " " + tok
        pretok_sent = pretok_sent[1:]
        return pretok_sent

    def get_sentences_nes_list(self, clinical_case):
        named_entities = []
        full_text_nes = self.ner_pipeline(clinical_case)
        current_entity = None
        current_text = ""
        for item in full_text_nes:
            # import ipdb;ipdb.set_trace()
            if item["entity"].startswith("B-"):
                # Start of a new entity
                if current_entity is not None:
                    # Append the previous entity to the current sentence
                    if len(current_text.strip()) > 1:
                        clean_text = self.detokenize_bert(current_text.strip())
                        named_entities.append(
                            (
                                current_entity,
                                clean_text,
                                current_start,
                                current_end,
                            )
                        )

                current_start = item["start"]
                current_entity = item["entity"][2:]
                current_text = item["word"]
                current_end = item["end"]

            elif item["entity"].startswith("I-"):
                # Continuation of the current entity
                current_text += " " + item["word"]
                current_end = item["end"]

        # Append the last entity to the current sentence
        if current_entity is not None:
            if len(current_text.strip()) > 1:
                clean_text = self.detokenize_bert(current_text.strip())
                named_entities.append(
                    (current_entity, clean_text, current_start, current_end)
                )

        return named_entities

    # def get_sentences_nes(self, sentences):
    #     sentences_nes = []
    #     full_text_nes = self.ner_pipeline(sentences)
    #     for entity_list in full_text_nes:
    #         #if len(entity_list) == 0:
    #             # If the entity_list is empty, append the original sentence
    #             #sentences_nes.append(sentences[full_text_nes.index(entity_list)])
    #             #continue

    #         current_sentence_nes = ""
    #         current_entity = None
    #         current_text = ""

    #         for item in entity_list:
    #             if item["entity"].startswith("B-"):
    #                 # Start of a new entity
    #                 if current_entity is not None:
    #                     # Append the previous entity to the current sentence
    #                     current_sentence_nes += " " + current_text.strip()

    #                 current_entity = item["entity"][2:]
    #                 current_text = item["word"]
    #             elif item["entity"].startswith("I-"):
    #                 # Continuation of the current entity
    #                 current_text += " " + item["word"]

    #         # Append the last entity to the current sentence
    #         if current_entity is not None:
    #             current_sentence_nes += " " + current_text.strip()

    #         # Append the current sentence to the list of sentences
    #         #detokenize the sentence before appending
    #         current_sentence_nes = self.detokenize_bert(current_sentence_nes)
    #         sentences_nes.append(current_sentence_nes.strip())

    #     return sentences_nes

    # def embed_sentences(self, sentences_nes):
    #     # turn sentences_nes into a list of strings
    #     string_sentences_nes = [' '.join(sentence_nes) for sentence_nes in sentences_nes]
    #     embedded_sentences_nes = self.sent_bert_model.encode(string_sentences_nes)

    #     # embedded_sentences_nes = [
    #     #    self.model(
    #     #        **self.tokenizer(sentence, return_tensors="pt").to(self.device)
    #     #    ).last_hidden_state.mean(dim=1).cpu().detach().numpy()
    #     #    for sentence in sentences_nes
    #     # ]
    #     return embedded_sentences_nes

    # def find_closest_sentence(self, answer_embedding, case_embeddings):
    #     # answer_embedding = answer_embedding.detach().numpy()
    #     cosine_similarities = [
    #         cosine_similarity(answer_embedding, embedding)
    #         for embedding in case_embeddings
    #     ]
    #     closest_sentence_index = np.argmax(cosine_similarities)

    #     return closest_sentence_index

    # def compute_distance_sentences(self, answer_embedding, case_embeddings):
    #     # answer_embedding = answer_embedding.cpu().detach().numpy()
    #     cosine_similarities = [
    #         # cosine_similarity(answer_embedding, embedding)
    #         dot(answer_embedding, embedding)
    #         / (norm(answer_embedding) * norm(embedding))
    #         for embedding in case_embeddings
    #     ]
    #     return cosine_similarities

    # def match_sentences_baseline(self, data, metric="jaccard"):
    #     # Uses jaccard distance to match sentences
    #     match_tuples = []
    #     already_generated = False

    #     # Check if dataframe already created and read it
    #     if os.path.exists(os.path.join(self.output_dir_dataframe, "symptoms.pkl")):
    #         symptoms_df = pd.read_pickle(os.path.join(self.output_dir_dataframe, "symptoms.pkl"))
    #         already_generated = True
    #     else:
    #         symptoms_df = pd.DataFrame(
    #             columns=[
    #                 "case_id",
    #                 "case_nes",
    #                 "answer_nes",
    #                 "answer_sentences",
    #                 "case_sentences",
    #                 "option_1",
    #                 "option_2",
    #                 "option_3",
    #                 "option_4",
    #                 "option_5",
    #                 "correct_answer",
    #                 "diagnosis_question",
    #                 "full_case",
    #                 "full_answer"
    #             ]
    #         )

    #     for index, row in tqdm(data.iterrows()):

    #         if already_generated:
    #             # get row by case_id
    #             saved_row = symptoms_df.loc[symptoms_df['case_id'] == row['line_id']]
    #             answer_sentence_nes = saved_row['answer_nes'].tolist()[0]
    #             case_sentences_nes = saved_row['case_nes'].tolist()[0]
    #             answer_sentences = saved_row['answer_sentences'].tolist()[0]
    #             case_sentences = saved_row['case_sentences'].tolist()[0]
    #         else:
    #             answer_sentences = self.split_sentences(row, "full_answer")
    #             case_sentences = self.split_sentences(row, "full_question")
    #             answer_sentence_nes = self.get_sentences_nes_list(answer_sentences)
    #             case_sentences_nes = self.get_sentences_nes_list(case_sentences)

    #             new_symptoms_row = {
    #                 "case_id": row["line_id"],
    #                 "case_nes": case_sentences_nes,
    #                 "answer_nes": answer_sentence_nes,
    #                 "case_sentences": case_sentences,
    #                 "answer_sentences": answer_sentences,
    #                 "option_1": row["option_1"],
    #                 "option_2": row["option_2"],
    #                 "option_3": row["option_3"],
    #                 "option_4": row["option_4"],
    #                 "option_5": row["option_5"],
    #                 "correct_answer": row["correct_answer"],
    #                 "diagnosis_question": row["diagnosis_question"],
    #                 "full_case": row["full_question"],
    #                 "full_answer": row["full_answer"]
    #             }
    #             symptoms_df = symptoms_df.append(new_symptoms_row, ignore_index=True)

    #         for i, answer_sentence in enumerate(answer_sentences):
    #             if answer_sentence_nes[i] == []:
    #                 continue
    #             for j, case_sentence in enumerate(case_sentences):
    #                 if case_sentences_nes[j] == []:
    #                     continue
    #                 if metric == "jaccard":
    #                     distance = self.compute_jaccard(case_sentence, answer_sentence)
    #                 elif metric == "fuzzy":
    #                     distance = self.compute_fuzzy_token_set_ratio(case_sentence, answer_sentence)
    #                     distance = distance / 100
    #                 if distance >= self.threshold:
    #                     match_tuples.append(
    #                         (
    #                             index,
    #                             answer_sentence,
    #                             case_sentence,
    #                             True,
    #                         )
    #                     )
    #                 else:
    #                     match_tuples.append(
    #                         (
    #                             index,
    #                             answer_sentence,
    #                             case_sentence,
    #                             False,
    #                         )
    #                     )

    #     return symptoms_df, match_tuples

    # def match_sentences(self, data):
    #     match_tuples = []
    #     already_generated = False

    #     # Check if dataframe already created and read it
    #     if os.path.exists(os.path.join(self.output_dir_dataframe, "symptoms.pkl")):
    #         symptoms_df = pd.read_pickle(os.path.join(self.output_dir_dataframe, "symptoms.pkl"))
    #         already_generated = True
    #     else:
    #         symptoms_df = pd.DataFrame(
    #             columns=[
    #                 "case_id",
    #                 "case_nes",
    #                 "answer_nes",
    #                 "answer_sentences",
    #                 "case_sentences",
    #                 "option_1",
    #                 "option_2",
    #                 "option_3",
    #                 "option_4",
    #                 "option_5",
    #                 "correct_answer",
    #                 "diagnosis_question",
    #                 "full_case",
    #                 "full_answer"
    #             ]
    #         )

    #     for index, row in tqdm(data.iterrows()):

    #         if already_generated:
    #             # get row by case_id
    #             saved_row = symptoms_df.loc[symptoms_df['case_id'] == row['line_id']]
    #             answer_sentence_nes = saved_row['answer_nes'].tolist()[0]
    #             case_sentences_nes = saved_row['case_nes'].tolist()[0]
    #             answer_sentences = saved_row['answer_sentences'].tolist()[0]
    #             case_sentences = saved_row['case_sentences'].tolist()[0]
    #         else:
    #             answer_sentences = self.split_sentences(row, "full_answer")
    #             case_sentences = self.split_sentences(row, "full_question")
    #             answer_sentence_nes = self.get_sentences_nes_list(answer_sentences)
    #             case_sentences_nes = self.get_sentences_nes_list(case_sentences)

    #             new_symptoms_row = {
    #                 "case_id": row["line_id"],
    #                 "case_nes": case_sentences_nes,
    #                 "answer_nes": answer_sentence_nes,
    #                 "case_sentences": case_sentences,
    #                 "answer_sentences": answer_sentences,
    #                 "option_1": row["option_1"],
    #                 "option_2": row["option_2"],
    #                 "option_3": row["option_3"],
    #                 "option_4": row["option_4"],
    #                 "option_5": row["option_5"],
    #                 "correct_answer": row["correct_answer"],
    #                 "diagnosis_question": row["diagnosis_question"],
    #                 "full_case": row["full_question"],
    #                 "full_answer": row["full_answer"]
    #             }
    #             symptoms_df = symptoms_df.append(new_symptoms_row, ignore_index=True)

    #         answer_embeddings = self.embed_sentences(answer_sentence_nes)
    #         case_embeddings = self.embed_sentences(case_sentences_nes)

    #         for i, answer_embedding in enumerate(answer_embeddings):
    #             cosine_distance = self.compute_distance_sentences(
    #                 answer_embedding, case_embeddings
    #             )

    #             for sentence_index in range(len(case_embeddings)):
    #                 if case_sentences_nes[sentence_index] == []:
    #                         continue
    #                 if answer_sentence_nes[i] == []:
    #                         continue
    #                 if cosine_distance[sentence_index] >= self.threshold:

    #                     match_tuples.append(
    #                         (
    #                             index,
    #                             answer_sentences[i],
    #                             case_sentences[sentence_index],
    #                             True,
    #                         )
    #                     )
    #                 else:
    #                     match_tuples.append(
    #                         (
    #                             index,
    #                             answer_sentences[i],
    #                             case_sentences[sentence_index],
    #                             False,
    #                         )
    #                     )

    #     return symptoms_df, match_tuples

    # def perform_ner(self, input_data):
    #     # symptoms_df, match_tuples = self.match_sentences_baseline(input_data, metric='fuzzy')
    #     symptoms_df, match_tuples = self.match_sentences(input_data)

    #     if not os.path.exists(self.output_dir):
    #         os.makedirs(self.output_dir)

    #     tuple_output_file = os.path.join(self.output_dir, "tuples.txt")
    #     with open(tuple_output_file, "w") as f:
    #         for t in match_tuples:
    #             f.write(f"{t}\n")

    #     symptoms_output_file = os.path.join(self.output_dir_dataframe, "symptoms.pkl")
    #     # symptoms_df.to_csv(symptoms_output_file)
    #     symptoms_df.to_pickle(symptoms_output_file)

    #     print(
    #         f"Experiment with transformer {self.transformer} and a threshold of {self.threshold} completed. Outputs saved in {self.output_dir}"
    #     )

    def clean_outputs_dir(self):
        shutil.rmtree(self.config.EXPERIMENTS_DIR)

    def get_tuple_files(self):
        base_dir = self.config.EXPERIMENTS_DIR
        return [
            os.path.join(base_dir, author, model_name, threshold, "tuples.txt")
            for author in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, author))
            for model_name in os.listdir(os.path.join(base_dir, author))
            if os.path.isdir(os.path.join(base_dir, author, model_name))
            for threshold in os.listdir(
                os.path.join(base_dir, author, model_name)
            )
            if os.path.isdir(
                os.path.join(base_dir, author, model_name, threshold)
            )
            and os.path.isfile(
                os.path.join(
                    base_dir, author, model_name, threshold, "tuples.txt"
                )
            )
        ]

    # def compute_f1(self, eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     report = classification_report(labels, predictions, output_dict=False)

    #     # ACAAAAAAAA
    #     report_path = os.path.join(
    #         self.output_dir, f"report_epoch_{self.current_epoch}.txt"
    #     )
    #     with open(report_path, "w") as f:
    #         f.write(report)

    #     self.current_epoch += 1

    #     # return metric.compute(predictions=predictions, references=labels, average="macro")
    #     return classification_report(labels, predictions, output_dict=True)["macro avg"]

    # def build_tuples_dataset(self, tuple_file):
    #     dataset = []
    #     with open(tuple_file, "r") as f:
    #         lines = f.readlines()
    #     for line in lines:
    #         case_no, expl_sent, case_sent, label = ast.literal_eval(line.strip())
    #         dataset.append(
    #             {
    #                 "case_id": int(case_no),
    #                 "label": int(label),
    #                 "explanation_sentence": expl_sent,
    #                 "case_sentence": case_sent,
    #             }
    #         )
    #     return dataset

    # def split_tuples_dataset(self, dataset):
    #     np.random.seed(42)
    #     train_indices = np.random.choice(
    #         self.config.total_size, self.config.train_size, replace=False
    #     )

    #     train_split = [el for el in dataset if el["case_id"] in train_indices]
    #     test_split = [el for el in dataset if el["case_id"] not in train_indices]

    #     final_dataset = {"train": train_split, "test": test_split}
    #     return final_dataset

    # def finetune_tuples(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"

    #     torch.manual_seed(self.config.random_seed)
    #     np.random.seed(self.config.random_seed)
    #     torch.manual_seed(self.config.random_seed)
    #     torch.cuda.manual_seed_all(self.config.random_seed)

    #     tuples_file = os.path.join(self.output_dir, "tuples.txt")
    #     tuples_dataset = self.build_tuples_dataset(tuples_file)
    #     split_dataset = self.split_tuples_dataset(tuples_dataset)

    #     tokenizer = AutoTokenizer.from_pretrained(self.transformer)

    #     tokenized_train_sentences = []
    #     for el in split_dataset["train"]:
    #         inputs = tokenizer(
    #             el["explanation_sentence"],
    #             el["case_sentence"],
    #             padding="max_length",
    #             truncation=True,
    #             max_length=256,
    #         )
    #         item = {key: torch.as_tensor(val) for key, val in inputs.items()}
    #         item["labels"] = torch.as_tensor(el["label"])
    #         keys = list(item.keys())
    #         for k in keys:
    #             item[k].to(device)

    #         tokenized_train_sentences.append(item)

    #     tokenized_test_sentences = []
    #     for el in split_dataset["test"]:
    #         inputs = tokenizer(
    #             el["explanation_sentence"],
    #             el["case_sentence"],
    #             padding="max_length",
    #             truncation=True,
    #             max_length=256,
    #         )
    #         item = {key: torch.as_tensor(val) for key, val in inputs.items()}
    #         item["labels"] = torch.as_tensor(el["label"])
    #         keys = list(item.keys())
    #         for k in keys:
    #             item[k].to(device)

    #         tokenized_test_sentences.append(item)

    #     # Load the model
    #     model = AutoModelForSequenceClassification.from_pretrained(self.transformer)
    #     model.to("cuda")
    #     # Set up training arguments
    #     output_model_dir = os.path.join(self.output_dir, "model")
    #     os.makedirs(output_model_dir, exist_ok=True)

    #     # Training arguments
    #     training_args = TrainingArguments(
    #         "scibert_trainer",
    #         evaluation_strategy="epoch",
    #         per_device_train_batch_size=self.config.per_device_train_batch_size,
    #         per_device_eval_batch_size=self.config.per_device_eval_batch_size,
    #         num_train_epochs=self.config.num_train_epochs,
    #         # learning_rate=2.5e-5,
    #         dataloader_pin_memory=False,
    #     )
    #     # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=5)

    #     # Set up trainer
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=tokenized_train_sentences,
    #         eval_dataset=tokenized_test_sentences,
    #         compute_metrics=self.compute_f1,
    #     )
    #     trainer.train()
    #     trainer.save_model(self.output_dir)

    """
    def finetune_ner_matches(self):
        tuple_files = self.get_tuple_files()
        for tuple_file in tuple_files:
    """
