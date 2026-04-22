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