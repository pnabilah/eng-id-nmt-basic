import pandas as pd
import tensorflow as tf
import yaml

def preprocess(filepath):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(filepath, sep='\t', header=None)
    df = df.sample(frac=1).reset_index(drop=True)                     # Shuffle
    sentences_en = df[1].values; sentences_id = df[3].values          # From data exploration
    # Tokenization and encoding
    max_tokens=config["model"]["max_tokens"]
    max_len = config["model"]["max_translation_len"]
    text_vec_layer_en = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=max_len)    # split by whitespace
    text_vec_layer_id = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=max_len)
    text_vec_layer_en.adapt(sentences_en)
    text_vec_layer_id.adapt([f"startofseq {s} endofseq" for s in sentences_id])
    # Split dataset
    X_train = tf.constant(sentences_en[:20_000])
    X_val = tf.constant(sentences_en[20_000:])
    X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_id[20_000:]])
    X_val_dec = tf.constant([f"startofseq {s}" for s in sentences_id[:20_000]])
    Y_train = tf.constant([f"{s} endofseq" for s in sentences_id[20_000:]])
    Y_val = tf.constant([f"{s} endofseq" for s in sentences_id[:20_000]])
    return X_train, X_val, X_train_dec, X_val_dec, Y_train, Y_val, text_vec_layer_en, text_vec_layer_id