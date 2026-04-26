import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def translate(model, sentence, max_length=100):
    model = load_model(model)
    vocab = model.get_layer("text_vectorization_1").get_vocabulary()
    translation = ""
    for word_idx in range(max_length):
        X = tf.constant([sentence])
        X_dec = tf.constant(["startofseq" + translation])
        y_proba = model.predict((X, X_dec))[0, word_idx]
        predicted_translated_word_id = np.argmax(y_proba)
        predicted_translated_word = vocab[predicted_translated_word_id]
        if predicted_translated_word == "endofseq":
            break
        translation += " " + predicted_translated_word
    return translation.strip()


def translate_beam_search(model, sentence, max_length=100, beam_width=10):
    model = load_model(model)
    vocab = model.get_layer("text_vectorization_1").get_vocabulary()
    X = tf.constant([sentence])
    best_candidates = [("startofseq", 0)]                               
    for word_idx in range(max_length):
        candidates = []
        for seq, score in best_candidates:
            if "endofseq" in seq:
                candidates.append((seq, score))
            else:
                X_dec = tf.constant([seq])
                y_proba = model.predict((X, X_dec))[0, word_idx]
                top_idx = np.argsort(y_proba)[-beam_width:]
                for idx in top_idx:
                    seq_candidate = seq + " " + vocab[idx]
                    score_candidate = np.log(y_proba[idx] + 1e-9) + score
                    candidates.append((seq_candidate, score_candidate))
            
        scores_candidates = [t[1] for t in candidates]
        sorted_scores = sorted(range(len(scores_candidates)), key=lambda i:scores_candidates[i])
        sorted_candidates = [candidates[i] for i in sorted_scores]
        best_candidates = sorted_candidates[-beam_width:]

        # candidates = sorted(candidates, key=lambda x: x[1])
        # best_candidates = candidates[-beam_width:]

        if all("endofseq" in seq for seq, _ in best_candidates):
            break

    return best_candidates[-1][0].replace("startofseq", "").replace("endofseq", "").strip()