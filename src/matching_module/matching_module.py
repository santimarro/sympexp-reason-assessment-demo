import pickle
import numpy as np
from flair.data import Sentence
from scipy.spatial import distance_matrix


# ------------------------------------------------------------------------------


def main(hpo_terms, input, embedding_full_eval):
    results = []
    method_full_eval = "sentences"

    # 4 --- Here we run the embeddings
    # if method_full_eval == "words":
    #    bests = match_words_with_context(gold_labels_data[gold_labels.iloc[0].disease_name],
    #                                     embedding_full_eval, threshold_
    #                                     full_eval)
    if method_full_eval == "sentences":
        bests = match_with_context(input, embedding_full_eval, hpo_terms)

        tmp = {
            "detected_symptoms_in_ner": input["detected_symptoms"],
            "predicted_matched_symptoms": bests,
        }
        results.append(tmp)
        # print("=====")
    return results


def match_with_context(case, embeddings, hpo_terms):
    five_bests_for_detected = []
    for i, detectedSymptom in enumerate(case["detected_symptoms"]):
        detectedSymptom = detectedSymptom.lower()
        # Split sentences of the case
        sentences = case["case"].split(". ")
        findInSentence = ""

        # Find the context of the detected symptom
        # ### V1
        for sentence in sentences:
            sentence += "."
            if detectedSymptom in sentence:
                findInSentence = sentence
                just_context = sentence.replace(detectedSymptom, "")
                break

        if findInSentence != "":
            just_context_sentence = Sentence(just_context)
            embeddings.embed(just_context_sentence)
            detected_symptom_sentence = Sentence(detectedSymptom)
            embeddings.embed(detected_symptom_sentence)
            # NO CONTEXT
            detected_context_embedding = combine_embeddings(
                just_context_sentence.embedding.cpu(),
                detected_symptom_sentence.embedding.cpu(),
            )

            hpo_term_names = list(hpo_terms.values())
            hpo_sentence_embeddings = np.array(
                [
                    term["sentence"].embedding.cpu().numpy()
                    for term in hpo_term_names
                ]
            )

            # Note: This assumes combine_embeddings can handle numpy arrays.
            combined_embeddings = combine_embeddings(
                just_context_sentence.embedding.cpu().numpy(),
                hpo_sentence_embeddings,
            )

            detected_context_embedding = np.array(
                [detected_context_embedding.numpy()]
            )

            # Calculate the distance matrix.
            dist = distance_matrix(
                detected_context_embedding, combined_embeddings
            )
            best = np.argsort(dist[0])[:5]
            bests = []
            list_hpo_context_embeddings_names = []
            for hpo_term in hpo_terms:
                list_hpo_context_embeddings_names.append(hpo_terms[hpo_term])
            for index in best:
                bests.append(
                    [dist[0][index], list_hpo_context_embeddings_names[index]]
                )
            five_bests = []
            for j, b in enumerate(bests):
                # print("For", detectedSymptom, "Bests", b[1]["name"])
                five_bests.append(b[1]["name"])
            # print("---")
            five_bests_for_detected.append(five_bests)

    return five_bests_for_detected


def combine_embeddings(embeddings1, embeddings2):
    return np.add(embeddings1, embeddings2)
