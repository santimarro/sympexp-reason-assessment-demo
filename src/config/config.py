import os


class Configuration:
    def __init__(self, transformer=None, threshold=None, epochs=None):
        # Pasar al main directamente - PARTES
        self.keep_only_symptoms = False
        self.run_ner = True
        self.run_onthology_matching = False
        self.run_matches_finetuning = True

        # Parameters
        self.API_SEARCH_URL = "https://hpo.jax.org/api/hpo/search"
        self.API_DISEASE_URL = "https://hpo.jax.org/api/hpo/disease/"
        self.threshold = threshold or 0.80
        self.transformer = (
            transformer
            or "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        self.num_train_epochs = epochs or 10
        self.embeddings_model = "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT"

        self.execution_method = "Single"
        self.random_seed = 42
        self.train_size = 497
        self.total_size = 621
        self.per_device_train_batch_size = 8
        self.per_device_eval_batch_size = 8
        self.auth_token = "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"

        self.auth_token = "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"
        # Directories
        self.BASE_DIR = self.BASE_DIR = os.getcwd()
        self.EXPERIMENTS_DIR = os.path.join(
            self.BASE_DIR, "outputs_Truco_2_fix"
        )
        self.DATASET_DIR = os.path.join(self.BASE_DIR, "casimedicos.csv")
        self.SYMPTOMS_FILE_DIR = os.path.join(self.BASE_DIR, "symptoms.csv")
        self.SETS_FILE_DIR = os.path.join(
            self.EXPERIMENTS_DIR, "symptoms_sets.csv"
        )

        self.HPO_EMBEDDINGS = os.path.join(
            self.BASE_DIR, "hpo_terms_embeddings_s_pubmedbert_ms_marco.pickle"
        )

        # Batch runs
        self.transformers_list = [
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "pritamdeka/S-PubMedBert-MS-MARCO",
            "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
        ]

        # self.thresholds_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        self.thresholds_list = [0.6]
