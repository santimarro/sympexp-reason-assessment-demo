import os
import shutil
import warnings

import torch
from transformers import pipeline

from src.config.config import Configuration

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

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.ner_pipeline = pipeline(
            "token-classification",
            model="smarro/medical_ner",
            use_auth_token=self.config.auth_token,
            device=self.device,
        )


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
        current_start = current_end = ""
        for item in full_text_nes:
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
            for threshold in os.listdir(os.path.join(base_dir, author, model_name))
            if os.path.isdir(os.path.join(base_dir, author, model_name, threshold))
            and os.path.isfile(
                os.path.join(base_dir, author, model_name, threshold, "tuples.txt")
            )
        ]
