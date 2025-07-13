# --- Imports ---
import os
import re
import json
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import DatasetDict

# Scikit-learn utilities
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# NLP preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sentence embedding and Word2Vec
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors

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

# Global variable to store label mapping
LABEL_MAPPING = {}

# --- Custom Transformers ---

class Word2VecVectorizer(TransformerMixin):
    """
    A custom scikit-learn compatible transformer for Word2Vec embeddings.
    It averages word vectors in each input sentence to produce a fixed-length representation.
    """
    def __init__(self, model):
        self.model = model
        self.dim = model.vector_size

    def transform(self, X):
        embeddings = []
        for sentence in X:
            tokens = sentence.split()
            vectors = [self.model[word] for word in tokens if word in self.model]
            avg_vector = np.mean(vectors, axis=0) if vectors else np.zeros(self.dim, dtype=np.float32)
            embeddings.append(avg_vector)
        return np.array(embeddings, dtype=np.float32)

    def fit(self, X, y=None):
        return self

class SentenceBERTVectorizer(TransformerMixin):
    """
    A scikit-learn compatible transformer for Sentence-BERT embeddings.
    Uses pre-trained transformer models to encode entire sentences.
    """
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def transform(self, X):
        return self.model.encode(X, convert_to_numpy=True)

    def fit(self, X, y=None):
        return self

# --- Preprocessing Functions ---

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
    global LABEL_MAPPING
    LABEL_MAPPING = {label: idx for idx, label in enumerate(encoder.classes_)}

    if verbose:
        print("Label Mapping:", LABEL_MAPPING)
        print(f"Processed {len(X_train)} training and {len(X_test)} test samples.")

    return X_train, X_test, y_train, y_test

# --- Model and Vectorizer Setup ---

def get_vectorizer(vectorizer_type, params, sentence_bert_model = None, word2vec_path=None, verbose=False):
    """
    Instantiate the appropriate vectorizer (BoW, TF-IDF, Word2Vec, Sentence-BERT).
    """
    if verbose:
        print(f"Initializing vectorizer: {vectorizer_type.upper()}")

    if vectorizer_type == "bow":
        return CountVectorizer(**params)
    elif vectorizer_type == "tfidf":
        return TfidfVectorizer(**params)
    elif vectorizer_type == "word2vec":
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        return Word2VecVectorizer(word2vec_model)
    elif vectorizer_type == "sentence_bert":
        return SentenceBERTVectorizer(model_name=sentence_bert_model)
    else:
        raise ValueError(f"Unknown vectorizer: {vectorizer_type}")

def get_classifier(clf_name, clf_params, verbose=False, multi_class=False):
    """
    Instantiate classifier and its hyperparameter grid.
    Supports Logistic Regression, ComplementNB, SVM, and Random Forest.
    """
    if verbose:
        print(f"Initializing classifier: {clf_name.upper()}")

    classifiers = {
        "lr": LogisticRegression(),
        "nb": ComplementNB(),
        "svm": SVC(decision_function_shape="ovr" if multi_class else "auto"),
        "rf": RandomForestClassifier()
    }

    param_grid = {
        "lr": {
            "classifier__C": clf_params["lr"]["C"],
            "classifier__solver": clf_params["lr"]["solver"],
            "classifier__max_iter": clf_params["lr"]["max_iter"]
        },
        "nb": {"classifier__alpha": clf_params["nb"]["alpha"]},
        "svm": {
            "classifier__C": clf_params["svm"]["C"],
            "classifier__kernel": clf_params["svm"]["kernel"]
        },
        "rf": {
            "classifier__n_estimators": clf_params["rf"]["n_estimators"],
            "classifier__max_depth": clf_params["rf"]["max_depth"],
            "classifier__min_samples_split": clf_params["rf"]["min_samples_split"]
        }
    }

    return classifiers[clf_name], param_grid[clf_name]

# --- Utility ---

def save_results(results, results_dir, filename):
    """
    Save classification results and best parameters to a JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved: {filepath}")

# --- Training Pipeline ---
def run_pipeline(X_train, X_test, y_train, y_test,
                 vectorizer_type, vectorizer_params,
                 clf_name, clf_params,
                 task_type, results_dir,
                 feature_selection_cfg=None,
                 word2vec_path=None,
                 sentence_bert_model=None,
                 verbose=False):
    """
    Full pipeline for training and evaluating a classifier using vectorization and optional feature selection.
    Includes grid search over hyperparameters.
    """
    multi_class = task_type == "cluster"
    vectorizer = get_vectorizer(vectorizer_type=vectorizer_type, 
                                params=vectorizer_params, 
                                word2vec_path=word2vec_path, 
                                sentence_bert_model=sentence_bert_model, 
                                verbose=verbose)
    classifier, param_grid = get_classifier(clf_name, clf_params, verbose, multi_class)

    steps = [("vectorizer", vectorizer)]

    # ---- CONDITIONAL FEATURE SELECTION ----
    if (
        feature_selection_cfg
        and feature_selection_cfg.get("enabled", False)
        and vectorizer_type in ["bow", "tfidf"]
        and clf_name in ["lr", "svm"]
    ):
        k_value = feature_selection_cfg.get("k", 1000)
        steps.append(("feature_selection", SelectKBest(score_func=mutual_info_classif, k=k_value)))
        if verbose:
            print(f"[✓] Feature selection enabled with k={k_value}")

    steps.append(("classifier", classifier))
    pipeline = Pipeline(steps)

    # ---- Construct Param Grid ----
    if vectorizer_type in ["bow", "tfidf"]:
        full_param_grid = {
            "vectorizer__max_features": vectorizer_params["max_features"],
            "vectorizer__min_df": vectorizer_params["min_df"],
            "vectorizer__ngram_range": [tuple(x) for x in vectorizer_params["ngram_range"]],
            **param_grid
        }
    else:
        full_param_grid = param_grid

    if verbose:
        print(f"\n[✓] Starting GridSearch for {clf_name.upper()} with {vectorizer_type.upper()}")
        print(json.dumps(full_param_grid, indent=2))

    print(X_train[:5])  # Debugging: print first 5 training samples
    # ---- Grid Search ----
    grid = GridSearchCV(
        pipeline,
        full_param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1,
        verbose=2 if verbose else 0
    )
    grid.fit(X_train, y_train)

    # ---- Evaluation ----
    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0 )

    results = {
        "best_params": grid.best_params_,
        "labels": LABEL_MAPPING,
        "classification_report": report
    }

    # ---- Save Results JSON ----
    json_filename = f"results_{vectorizer_type}_{clf_name}_{task_type}.json"
    save_results(results, results_dir, json_filename)

    # ---- Save Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(LABEL_MAPPING.keys()),
                yticklabels=list(LABEL_MAPPING.keys()))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_path = os.path.join(results_dir, f"confusion_matrix_{vectorizer_type}_{clf_name}_{task_type}.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    if verbose:
        print(f"[✓] Confusion matrix saved at: {cm_path}")

    # ---- Output Summary ----
    print("\nBest Parameters:", json.dumps(grid.best_params_, indent=2))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return grid.best_estimator_, grid.best_params_