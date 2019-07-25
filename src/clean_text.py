# Import required libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

# Prerequisites for cleaning
nltk.download("stopwords")                      # Download stopwords from NLTK library
nltk.download('wordnet')                        # Download wordnet, a lexixal database from NLTK library
stopwords = set(stopwords.words('english'))     # Store stopwords
lemmatizer = WordNetLemmatizer()                # Create object for lemmatization

# Function for standard cleaning of text (remove punctuations, abbreviations, etc.) using regular expressions
def standard_clean(text):
  text = str(text)
  text = text.lower()
  text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "cannot ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r",", " ", text)
  text = re.sub(r"\.", " ", text)
  text = re.sub(r"!", " ! ", text)
  text = re.sub(r"\/", " ", text)
  text = re.sub(r"\^", " ^ ", text)
  text = re.sub(r"\+", " + ", text)
  text = re.sub(r"\-", " - ", text)
  text = re.sub(r"\=", " = ", text)
  text = re.sub(r"'", " ", text)
  text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
  text = re.sub(r":", " : ", text)
  text = re.sub(r" e g ", " eg ", text)
  text = re.sub(r" b g ", " bg ", text)
  text = re.sub(r" u s ", " american ", text)
  text = re.sub(r"\0s", "0", text)
  text = re.sub(r" 9 11 ", "911", text)
  text = re.sub(r"e - mail", "email", text)
  text = re.sub(r"j k", "jk", text)
  text = re.sub(r"\s{2,}", " ", text)
  return text

# Function to remove stopwords from a sentence
def remove_stopwords(text):
  text = text.split()
  clean = ""
  for w in text:
    if w not in stopwords:
      clean = clean + " " + w
  return str(clean[1:])

# Function to lemmatize words of a sentence using Lemmatizer object
def lemmatize(text):
  text = text.split()
  clean = ""
  for w in text:
    clean = clean + " " + lemmatizer.lemmatize(w)
  return str(clean[1:])

# Function to clean the text
def clean(text):
  text = standard_clean(text)
  text = remove_stopwords(text)
  text = lemmatize(text)
  return text
