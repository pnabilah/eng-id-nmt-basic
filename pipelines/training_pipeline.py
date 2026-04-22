import yaml
from eng_id_nmt_basic.preprocess import preprocess
from eng_id_nmt_basic.models import build_seq2seq_lstm_model
from eng_id_nmt_basic.train import train_seq2seq_model
from eng_id_nmt_basic.utils.visualization import plot_loss_history, plot_accuracy_history

def training_pipeline():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Preprocess data
    X_train, X_val, X_train_dec, X_val_dec, Y_train, Y_val, text_vec_layer_en, text_vec_layer_id = preprocess(filepath=config["dataset"]["raw_path"])
    # Build model
    model = build_seq2seq_lstm_model(text_vec_layer_in=text_vec_layer_en,
                                     text_vec_layer_tar=text_vec_layer_id,
                                     vocab_size=config["model"]["max_tokens"],
                                     embed_dim=config["model"]["embed_dim"],
                                     lstm_units=config["model"]["lstm_units"])
    # Train model
    history = train_seq2seq_model(model, X_train=(X_train, X_train_dec), Y_train=Y_train, 
                                  X_val=(X_val, X_val_dec), Y_val=Y_val,
                                  epochs=config["training"]["epochs"],
                                  best_model_path=config["paths"]["best_model_path"],
                                  final_model_path=config["paths"]["final_model_path"],
                                  history_path=config["paths"]["history_path"])
    # Plot metrics
    plot_loss_history(history, save_path=config["paths"]["loss_curve_path"])
    plot_accuracy_history(history, save_path=config["paths"]["accuracy_curve_path"])

if __name__ == "__main__":
    training_pipeline()