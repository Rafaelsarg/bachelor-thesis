import re
import string
from datasets import DatasetDict
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# --- Global Configuration ---
STOPWORDS = set(stopwords.words("english"))  # Set of English stopwords
LEMMATIZER = WordNetLemmatizer()  # Initialize a lemmatizer for reducing words to their base forms

def clean_text_bow_tfidf(text):
    """
    Cleaning pipeline for BoW and TF-IDF models.
    Lowercasing, digit and punctuation removal, tokenization, stopword removal, and lemmatization.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def clean_text_embeddings(text):
    """
    Minimal cleaning for embedding-based models (Word2Vec, Sentence-BERT).
    Only lowercasing is applied to retain semantic structure.
    """
    return text.lower()

def process_struggle_response_pair(text: str):
    """
    Extracts 'struggle' and 'response' from text using regex. Expected format: <struggle> #label# <response> #END#
    """
    match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", text)
    if not match:
        return None
    struggle = match.group(1).strip()
    response = match.group(3).strip()
    return f"Struggle: {struggle} Response: {response}"

def preprocess_data(dataset_path: str, vectorizer_type: str, task_type: str, verbose=False):
    """
    Load and preprocess the dataset for classification.
    Selects preprocessing based on vectorizer and handles both binary and multi-class labels.
    """
    dataset = DatasetDict.load_from_disk(dataset_path)

    # Choose the correct cleaning function
    preprocess_fn = clean_text_bow_tfidf if vectorizer_type in ["bow", "tfidf"] else clean_text_embeddings
    extract_fn = process_struggle_response_pair if task_type == "safety" else lambda x: x

    # Apply extraction + preprocessing
    X_train = [preprocess_fn(extract_fn(x)) for x in dataset["train"]["struggle"] if extract_fn(x)]
    X_test = [preprocess_fn(extract_fn(x)) for x in dataset["test"]["struggle"] if extract_fn(x)]
    y_train = dataset["train"]["label"]
    y_test = dataset["test"]["label"]

    # Convert N/Y labels to Unsafe/Safe for safety task
    if task_type == "safety":
        label_map = {'N': 'Unsafe', 'Y': 'Safe'}
        y_train = [label_map[l] for l in y_train]
        y_test = [label_map[l] for l in y_test]

    # Encode string labels as integers
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Save label mapping for interpretability
    label_mapping = {label: idx for idx, label in enumerate(encoder.classes_)}

    if verbose:
        print("Label Mapping:", label_mapping)
        print(f"Processed {len(X_train)} training and {len(X_test)} test samples.")

    return X_train, X_test, y_train, y_test, label_mapping
