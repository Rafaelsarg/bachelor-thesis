# --- Imports ---
import os
import re
import json
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import DatasetDict

# Local imports
from baselines.utils import save_results
from baselines.data_processor import clean_text_bow_tfidf, clean_text_embeddings, process_struggle_response_pair, preprocess_data

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

# Sentence embedding and Word2Vec
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors


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
        "svm": SVC(decision_function_shape="ovr" if multi_class else "ovo"),
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

def run_pipeline(
    X_train,
    X_test,
    y_train,
    y_test,
    label_mapping,
    vectorizer_type: str,
    vectorizer_params: dict,
    clf_name: str,
    clf_params: dict,
    task_type: str,
    results_dir: str,
    feature_selection_cfg: dict = None,
    word2vec_path: str = None,
    sentence_bert_model: str = None,
    verbose: bool = False
) -> tuple:
    """Execute the full classification pipeline with vectorization, training, and evaluation.

    This function constructs a scikit-learn pipeline with a vectorizer, optional feature selection,
    and a classifier, performs grid search for hyperparameter optimization, evaluates the model
    on the test set, and saves results as JSON and a confusion matrix plot.

    Args:
        X_train (list): Training text data.
        X_test (list): Test text data.
        y_train (list): Training labels (encoded as integers).
        y_test (list): Test labels (encoded as integers).
        vectorizer_type (str): Type of vectorizer ("bow", "tfidf", "word2vec", "sentence_bert").
        vectorizer_params (dict): Parameters for BoW or TF-IDF vectorizers.
        clf_name (str): Classifier name ("lr", "nb", "svm", "rf").
        clf_params (dict): Hyperparameters for grid search.
        task_type (str): Task type ("safety" for binary, "cluster" for multi-class).
        results_dir (str): Directory to save results and plots.
        feature_selection_cfg (dict, optional): Configuration for feature selection.
        word2vec_path (str, optional): Path to Word2Vec model file.
        sentence_bert_model (str, optional): Sentence-BERT model name.
        verbose (bool): If True, print detailed logs.

    Returns:
        tuple: (best_estimator, best_params) where best_estimator is the trained model and
               best_params is a dictionary of the best hyperparameters found.
    """
    # Validate input parameters
    valid_vectorizers = {"bow", "tfidf", "word2vec", "sentence_bert"}
    valid_classifiers = {"lr", "nb", "svm", "rf"}
    valid_tasks = {"safety", "cluster"}
    if vectorizer_type not in valid_vectorizers:
        raise ValueError(f"Invalid vectorizer_type: {vectorizer_type}. Choose from {valid_vectorizers}")
    if clf_name not in valid_classifiers:
        raise ValueError(f"Invalid clf_name: {clf_name}. Choose from {valid_classifiers}")
    if task_type not in valid_tasks:
        raise ValueError(f"Invalid task_type: {task_type}. Choose from {valid_tasks}")
    if vectorizer_type in ["word2vec", "sentence_bert"] and clf_name == 'nb':
        raise ValueError("ComplementNB is not compatible with Word2Vec or Sentence-BERT vectorizers.")

    # Determine if task is multi-class
    is_multi_class = task_type == "cluster"

    # Initialize vectorizer and classifier
    vectorizer = get_vectorizer(
        vectorizer_type=vectorizer_type,
        params=vectorizer_params,
        word2vec_path=word2vec_path,
        sentence_bert_model=sentence_bert_model,
        verbose=verbose
    )
    classifier, param_grid = get_classifier(
        clf_name=clf_name,
        clf_params=clf_params,
        verbose=verbose,
        multi_class=is_multi_class
    )

    # Construct pipeline
    pipeline_steps = [("vectorizer", vectorizer)]
    if (
        feature_selection_cfg
        and feature_selection_cfg.get("enabled", False)
        and vectorizer_type in ["bow", "tfidf"]
        and clf_name in ["lr", "svm"]
    ):
        k_features = feature_selection_cfg.get("k", 1000)
        pipeline_steps.append(("feature_selection", SelectKBest(score_func=mutual_info_classif, k=k_features)))
        if verbose:
            print(f"[✓] Enabled feature selection with k={k_features} features")

    pipeline_steps.append(("classifier", classifier))
    pipeline = Pipeline(pipeline_steps)

    # Build parameter grid for grid search
    if vectorizer_type in ["bow", "tfidf"]:
        full_param_grid = {
            "vectorizer__max_features": vectorizer_params["max_features"],
            "vectorizer__min_df": vectorizer_params["min_df"],
            "vectorizer__ngram_range": [tuple(x) for x in vectorizer_params["ngram_range"]],
            **param_grid
        }
    else:
        full_param_grid = param_grid

    # Log grid search details if verbose
    if verbose:
        print(f"\n[✓] Starting GridSearch for {clf_name.upper()} with {vectorizer_type.upper()}")
        print(json.dumps(full_param_grid, indent=2))
        print(f"First 5 training samples: {X_train[:5]}")

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        full_param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1,
        verbose=2 if verbose else 0
    )
    grid_search.fit(X_train, y_train)

    # Evaluate model on test set
    predictions = grid_search.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # Store results with label mapping from preprocess_data
    results = {
        "best_params": grid_search.best_params_,
        "labels": label_mapping,
        "classification_report": report
    }

    # Save results to JSON
    json_filename = f"results_{vectorizer_type}_{clf_name}_{task_type}.json"
    save_results(results, results_dir, json_filename)

    # Generate and save confusion matrix plot
    confusion_mat = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(label_mapping.keys()),
        yticklabels=list(label_mapping.keys())
    )
    plt.title(f"Confusion Matrix ({vectorizer_type.upper()} - {clf_name.upper()})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    confusion_matrix_path = os.path.join(results_dir, f"confusion_matrix_{vectorizer_type}_{clf_name}_{task_type}.png")

    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
    if verbose:
        print(f"[✓] Saved confusion matrix at: {confusion_matrix_path}")

    # Output results summary
    print("\nBest Parameters:", json.dumps(grid_search.best_params_, indent=2))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return grid_search.best_estimator_, grid_search.best_params_