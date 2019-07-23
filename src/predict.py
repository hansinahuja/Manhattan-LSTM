# Import required libraries
import tensorflow as tf
import pandas as pd

# Import local modules
import preprocess_data as pp
from create_model import manh_lstm_distance

# File paths
CHECKPOINT_PATH = '../checkpoints'
MODEL_FILE_NAME = '/model.h5'
DATASET_PATH = '../dataset'
TEST_FILE_NAME = '/test.csv'

# Prediction parameters
skiprows = 0            # Predict labels for question pairs from index 'skiprows'
nrows = 500             # to index 'skiprows + nrows' in the test file

threshold = 0.1         # Minimum Manhattan LSTM distance between two outputs
                        # for them to be classified as semantically similar


# Load trained model
print("Loading model...")
model = tf.keras.models.load_model(CHECKPOINT_PATH + MODEL_FILE_NAME, custom_objects={"manh_lstm_distance": manh_lstm_distance})

# Read test file
print("Reading test data...")
df = pd.read_csv(DATASET_PATH + TEST_FILE_NAME, skiprows=skiprows, nrows=nrows)

# Preprocess test data
print("Preprocessing test data...")
question1, question2 = pp.preprocess(df, mode='predict')

# Predict Manhattan LSTM distances
print("Predicting Manhattan LSTM distances...")
manh_lstm_distance = model.predict([question1, question2], verbose=1)

# Make binary predictions
print("Making binary predictions...")
prediction = manh_lstm_distance>threshold
prediction = prediction.astype(int)

# Print predictions
data = {'Manhattan LSTM distances': list(manh_lstm_distance), 'Prediction': list(prediction)}
df = pd.DataFrame(data)
print(df)


