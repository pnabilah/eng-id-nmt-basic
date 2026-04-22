import tensorflow as tf

def build_seq2seq_lstm_model(text_vec_layer_in, text_vec_layer_tar, vocab_size, embed_dim=128, lstm_units=512):
    # Input layer
    encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    # Text vectorization layer
    encoder_input_ids = text_vec_layer_in(encoder_inputs)
    decoder_input_ids = text_vec_layer_tar(decoder_inputs)
    # Embedding layer
    encoder_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    decoder_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    encoder_embedding = encoder_embedding_layer(encoder_input_ids)
    decoder_embedding = decoder_embedding_layer(decoder_input_ids)
    # Encoder
    encoder = tf.keras.layers.LSTM(units=lstm_units, return_state=True)
    last_seq_en_output, final_memory_state_en, final_carry_state_en = encoder(encoder_embedding)
    # Decoder
    decoder = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)
    whole_seq_dec_output = decoder(decoder_embedding, initial_state=[final_memory_state_en, final_carry_state_en])
    # Dense layer
    output_layer = tf.keras.layers.Dense(units=vocab_size, activation="softmax")
    Y_proba = output_layer(whole_seq_dec_output)
    # Create model
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba])
    model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model