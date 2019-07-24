# manhattan-lstm

Implementing Manhattan LSTM, a Siamese deep network to predict sentence to sentence semantic similarity.

Implementation inspired by [this paper](https://www.semanticscholar.org/paper/Siamese-Recurrent-Architectures-for-Learning-Mueller-Thyagarajan/6812fb9ef1c2dad497684a9020d8292041a639ff) by Mueller & Thyagarajan, and this [this Medium article](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07) by Elior Cohen.

## A few links

- Original research paper: [Siamese Recurrent Architectures for Learning Sentence Similarity](https://www.semanticscholar.org/paper/Siamese-Recurrent-Architectures-for-Learning-Mueller-Thyagarajan/6812fb9ef1c2dad497684a9020d8292041a639ff)
- The dataset can be found in `./dataset` or here: [First Quora Dataset Release: Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- Pretrained GloVe vectors can be found in `./pretrained_embeddings` or here: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

## Description

The detailed explanation of the model can be found in the aforementioned paper. The model described in the paper:

![MaLSTM model](img/malstm.JPG)

A ~2% increase in accuracy was observed compared to the aforementioned article on making a few changes and fine-tuning the model. The major ones are as follows:
- Cleaning the data was done using regular expressions, stopwords and PorterStemmer object from the NLTK library.
- The embedding matrix was created using pretrained GloVe vectors.
- The Embedding layer was made trainable.
- Adam optimizer was used with a gradient clipping norm = 1.5.
- Binary crossentropy loss was used.

Visualization of model returned by `./src/create_model.py`:

![Model plot](img/model_plot.png)
