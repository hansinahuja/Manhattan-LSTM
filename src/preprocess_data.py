# Import required libraries
import tensorflow as tf
import numpy as np
import pickle           # Library to load Tokenizer object

# Import local modules
import clean_text as ct

# Preprocessing parameters
maxlen=15        # Maximum number of words in a processed question


# Function to create Tokenizer object
def tokenize(df):
    df['concatenated'] = df['question1'] + " " + df['question2']
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.concatenated)
    return tokenizer

# Function to preprocess textual data
def preprocess(df, mode):

    # Clean the text
    df['question1'] = df.question1.map(lambda x: ct.clean(x))
    df['question2'] = df.question2.map(lambda x: ct.clean(x))

    # Prepare the data for the model
    print("Preparing data for model...")

    # While training, create Tokenizer object and also return labels
    if mode=='train':
        tokenizer = tokenize(df)
        df['question1'] = tokenizer.texts_to_sequences(df.question1)
        df['question2'] = tokenizer.texts_to_sequences(df.question2)
        question1 = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(df.question1, maxlen=maxlen)))
        question2 = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(df.question2, maxlen=maxlen)))
        labels = np.array(list(df.is_duplicate))
        return question1, question2, labels, tokenizer

    # While predicting, load existing Tokenizer object
    if mode=='predict':
        with open('../checkpoints/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        df['question1'] = tokenizer.texts_to_sequences(df.question1)
        df['question2'] = tokenizer.texts_to_sequences(df.question2)
        question1 = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(df.question1, maxlen=maxlen)))
        question2 = np.array(list(tf.keras.preprocessing.sequence.pad_sequences(df.question2, maxlen=maxlen)))
        return question1, question2


