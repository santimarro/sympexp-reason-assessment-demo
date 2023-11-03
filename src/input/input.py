import pandas as pd
import re

from src.config.config import Configuration


class InputModule:
    def __init__(self):
        self.config = Configuration()

    def replace_n_answer(self, row):
        answer_pattern = r"(?i)answer(\s+is)?\s+(\d+)"
        option_pattern = r"(?i)option\s+(\d+)"
        text = row["full_answer"]

        def replace_answer(match):
            answer_column = "option_" + match.group(2)
            if answer_column in row.index:
                answer_value = row[answer_column]
                answer_value = answer_value.rstrip(
                    "."
                )  # Remove trailing '.' character
                return match.group().replace(match.group(2), answer_value)
            return match.group()

        def replace_option(match):
            option_column = "option_" + match.group(1)
            if option_column in row.index:
                option_value = row[option_column]
                return option_value.rstrip(
                    "."
                )  # Remove trailing '.' character
            return match.group()

        replaced_text = re.sub(answer_pattern, replace_answer, text)
        replaced_text = re.sub(option_pattern, replace_option, replaced_text)
        row["full_answer"] = replaced_text
        return row

    def read_input_file(self):
        # TO DO: optimize loaded columns
        input_data = pd.read_csv(self.config.DATASET_DIR)
        input_data = input_data.apply(self.replace_n_answer, axis=1)
        return input_data
