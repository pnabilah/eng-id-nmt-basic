# English–Indonesian Neural Machine Translation

A basic English-to-Indonesian neural machine translation (NMT) system built with TensorFlow/Keras using an LSTM encoder–decoder architecture.

## Model

* Word-level tokenization
* LSTM encoder–decoder architecture
* 128-dimensional word embeddings
* 512 LSTM units
* Greedy decoding
* Beam search decoding

## Dataset

The model is trained on **25,331 English–Indonesian sentence pairs** from the [Tatoeba dataset](https://tatoeba.org/en/downloads).

## Example Translations

The model is still a basic implementation and translation quality is currently limited by the relatively small dataset and model architecture.

| English                  | Model Output                  |
| ------------------------ | ----------------------------- |
| I do not like movies     | aku tidak suka film           |
| She is cooking some food | dia sedang bermain sepak bola |
| We will go to the zoo    | kita akan pergi ke [UNK]      |

## Project Structure

The project is organized into separate components for preprocessing, training, and inference, with model and training settings managed through a `config.yaml` file.

```text
eng-id-nmt-basic/
├── config.yaml
├── data/
│   └── English-Indonesian.tsv
├── models/
│   └── seq2seqlstm1/
│       ├── best_val_model.keras
│       ├── final_epoch_model.keras
│       └── training_history.json
├── pipelines/
│   ├── inference_pipeline.ipynb
│   └── training_pipeline.py
├── results/
│   ├── seq2seqlstm1_accuracy_curve.png
│   └── seq2seqlstm1_loss_curve.png
├── src/
│   └── eng_id_nmt_basic/
│       ├── models.py
│       ├── preprocess.py
│       ├── train.py
│       ├── translate.py
│       └── utils/
│           ├── helper.py
│           └── visualization.py
├── pyproject.toml
└── README.md
```

## Future Improvements

* [ ] Add BLEU score evaluation
* [ ] Implement attention-based encoder–decoder
* [ ] Experiment with Transformer architecture
* [ ] Train on a larger English–Indonesian dataset
* [ ] Improve handling of unknown words (`[UNK]`)

## Reference

Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.

The implementation is based on the book's neural machine translation example and has been adapted and extended for English–Indonesian translation.