# Import required libraries
import tensorflow as tf
import numpy as np

# File paths
EMBEDDING_FILE_PATH = '../pretrained_embeddings/glove.6B.100d.txt'

# Model parameters
maxlen = 15             # Maximum number of words in a sentence
n_units = 50            # Number of units in LSTM layer
clipnorm = 1.5          # Norm for gradient clipping
EMBEDDING_DIM = 100     # Dimension of embedding vectors


# Function to calculate Manhattan LSTM distance
def manh_lstm_distance(question1, question2):
  distance = tf.keras.backend.abs(question1-question2)
  distance = tf.keras.backend.sum(distance, axis=1, keepdims=True)
  distance = -distance
  distance = tf.keras.backend.exp(distance)
  return distance

# Function to create Embedding layer using Tokenizer object
def create_embedding_layer(tokenizer):

    # Create a dictionary mapping vocabulary words to embedding vectors
    embeddings_index = {}
    f = open(EMBEDDING_FILE_PATH, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # Create embedding matrix to initialize Embedding layer
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Create Embedding layer
    embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix)
    embedding_layer = tf.keras.layers.Embedding(len(word_index)+1,
                                                EMBEDDING_DIM,
                                                embeddings_initializer=embeddings_initializer,
                                                input_length=maxlen,
                                                trainable=True,
                                                name="embedding_layer")

    return embedding_layer

# Function to create Siamese Manhattan LSTM model
def create(tokenizer):
    print("Creating model...")
    embedding_layer = create_embedding_layer(tokenizer)
    question1 = tf.keras.layers.Input(shape=(maxlen,), dtype='int32', name="question1")
    question2 = tf.keras.layers.Input(shape=(maxlen,), dtype='int32', name="question2")
    question1_encoded = embedding_layer(question1)
    question2_encoded = embedding_layer(question2)
    common_lstm_layer = tf.keras.layers.LSTM(n_units, name="common_lstm_layer")
    question1_output = common_lstm_layer(question1_encoded)
    question2_output = common_lstm_layer(question2_encoded)
    manhattan_lstm_distance = tf.keras.layers.Lambda(lambda x: manh_lstm_distance(x[0], x[1]), name="manhattan_lstm_distance")([question1_output, question2_output])
    model = tf.keras.models.Model([question1, question2], manhattan_lstm_distance)
    loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(clipnorm=clipnorm)
    metrics = ['accuracy', 'mse']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
