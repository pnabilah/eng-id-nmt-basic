import matplotlib.pyplot as plt
from eng_id_nmt_basic.utils.helper import prepare_save_path

def plot_loss_history(history, save_path=None):
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Training Loss')
    ax.plot(history['val_loss'], label ='Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    if save_path is not None:
        save_path = prepare_save_path(save_path)
        fig.savefig(save_path)        
        print(f"Image has been saved to {save_path}")       
    plt.show()

def plot_accuracy_history(history, save_path=None):
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='Training Accuracy')
    ax.plot(history['val_accuracy'], label='Validation Accuracy')
    ax.set_title('Training vs Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    if save_path is not None:
        save_path = prepare_save_path(save_path)
        fig.savefig(save_path)        
        print(f"Image has been saved to {save_path}")      
    plt.show()