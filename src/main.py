# Library to save Tokenizer object
import pickle

# Import local modules
import load_data as ld
import create_model as cm

# File paths
CHECKPOINT_PATH = '../checkpoints'
WEIGHTS_FILE_NAME = '/weights'
MODEL_FILE_NAME = '/model.h5'
TOKENIZER_FILE_NAME = '/tokenizer.pickle'

# Training parameters
epochs = 2                  # Number of epochs
batch_size = 64             # Training batch size
validation_split=0.2        # Fraction of training data for validation
verbose=1                   # Show progress bar

# Load, process training data and create Tokenizer onject
question1, question2, labels, tokenizer = ld.load()

# Create model using Tokenizer object
model = cm.create(tokenizer)

# Train the model
print("Training model...")
model.fit([question1, question2], labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

# Save model and model weights
print("Saving model and model weights...")
model.save(CHECKPOINT_PATH + MODEL_FILE_NAME)
model.save_weights(CHECKPOINT_PATH + WEIGHTS_FILE_NAME)

# Save Tokenizer object
print("Saving Tokenizer object...")
with open(CHECKPOINT_PATH + TOKENIZER_FILE_NAME, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
