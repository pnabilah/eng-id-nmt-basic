import json
from eng_id_nmt_basic.utils.helper import prepare_save_path
from tensorflow.keras.callbacks import ModelCheckpoint

def train_seq2seq_model(model, X_train:tuple, Y_train, X_val:tuple|None=None, Y_val=None, epochs=10,
                        best_model_path="models/best_val_model.keras",
                        final_model_path="models/final_epoch_model.keras",
                        history_path="models/training_history.json"):
    
    best_model_path = prepare_save_path(best_model_path)
    final_model_path = prepare_save_path(final_model_path)
    history_path = prepare_save_path(history_path)

    callbacks = []

    if X_val is not None and Y_val is not None:
        checkpoint = ModelCheckpoint(best_model_path, 
                                    monitor="val_accuracy",
                                    save_best_only=True)
        callbacks.append(checkpoint)
    
    history = model.fit(X_train, Y_train, 
                        validation_data=(X_val, Y_val), 
                        epochs=epochs, 
                        callbacks=callbacks)
    
    model.save(final_model_path)
    
    with open(history_path, "w") as f:
        json.dump(history.history, f)
        
    print("Model and training history has been saved.")
    return history.history