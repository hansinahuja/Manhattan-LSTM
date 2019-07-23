# Import required libraries
import pandas as pd

# Import local modules
import preprocess_data as pp

# File paths
DATASET_PATH = '../dataset'
TRAIN_FILE_NAME = '/train.csv'

# Function to load, process training data and create Tokenizer onject
def load():
    print("Reading training data...")
    df = pd.read_csv(DATASET_PATH + TRAIN_FILE_NAME)

    print("Preprocessing training data...")
    question1, question2, labels, tokenizer = pp.preprocess(df, mode='train')

    return question1, question2, labels, tokenizer
