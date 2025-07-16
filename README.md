# Thesis Codebase: Dietary Topic and Safety Classification

This repository contains the codebase developed for thesis experiments involving dietary topic classification and safety assessment of AI-generated nutrition counseling.


# 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Running Baseline Experiments](#running-baseline-experiments)
- [Baseline Configuration Explained](#baseline-configuration-explained)
- [Baseline Classification Module](#baseline-classification-module)
- [Running Prompting Experiments](#running-prompting-experiments)
- [Prompting Configuration Explained](#prompting-configuration-explained)
- [Prompting Classification Module](#prompting-classification-module)


# Prerequisites

To run the codebase successfully, ensure that the following software and accounts are available and properly configured:

- **Python 3.12.3**  
  The project used Python version 3.12.3 throughout experiments. It is recommended to use a virtual environment to avoid conflicts with other packages.

- **NVIDIA GPU: A40 or A100 (tested on Linux machines)**  
  Training and fine-tuning large language models is computationally intensive. The code was tested on Linux machines equipped with NVIDIA A40 and A100 40GB or 80GB GPUs. Other CUDA-compatible GPUs may also work, but performance and compatibility are not guaranteed.

- **CUDA-compatible GPU drivers**  
  Ensure that the correct NVIDIA drivers and CUDA toolkit are installed on your system. These are necessary for PyTorch and Hugging Face Transformers to leverage GPU acceleration.

- **[Hugging Face](https://huggingface.co) account and access token**  
  Required to download and use pre-trained models from the Hugging Face Hub (e.g., Mistral, Phi, LLaMA).  
  A token will be provided for reproduction purposes. Alternatively, you can [sign up for a free Hugging Face account](https://huggingface.co/join) and [generate your own access token](https://huggingface.co/settings/tokens) with at least `read` permission.

  ⚠️ **For LLaMA model** (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`), you must first request access from Meta. Visit the [LLaMA model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and click **"Access repository"**. You must be logged in to your Hugging Face account. Approval from Meta is required before you can download or use the model.

- **[Weights & Biases](https://wandb.ai/) account and API key**  
  Used for experiment tracking and visualization. The code automatically logs metrics such as loss, accuracy, and F1-score, along with configuration and model parameters.

  You will need to **create a Weights & Biases account** to enable logging. Although the platform is paid, it offers **educational access** — you can [request academic access here](https://wandb.ai/site/education).

  If you prefer to use your own account:
  - Register at [wandb.ai](https://wandb.ai/)
  - Generate your own API token from [your account settings](https://wandb.ai/authorize)
  - A detailed explanation of how to change the script to use your own account will be provided.

  My own Weights & Biases code will be included in the repository to ensure reproducibility. However, **you won't be able to view my hosted dashboards**.  
  If you don’t want to use W&B, you can simply **refuse logging during the first run** — the code will proceed without saving metrics to the platform.


- **[Ollama](https://ollama.com/)**  
  Required for prompting tasks that use locally hosted LLMs (e.g., Mistral, LLaMA, Phi). Install Ollama following the instructions on their website, and make sure it is running before launching any prompting scripts.


---

# Project Structure

Below is an overview of the main folders and their roles:


```
bachelor-thesis/
├── scripts/                    # Main execution scripts
│   ├── run_finetuning.py      # Fine-tuning experiments
│   ├── run_inference.py       # Model inference and 
│   ├── run_prompting.py       # Prompting experiments
│   └── run_baselines.py       # Baseline classification
├── src/                       # Source code modules
│   ├── finetuning/           # Fine-tuning implementation
│   ├── prompting/            # Prompting implementation
│   └── baselines/            # Baseline classification
├── config/                    # Configuration files
│   ├── finetune_config.yaml  # Fine-tuning configuration
│   ├── inference_config.yaml # Inference configuration
│   ├── prompting_config.yaml # Prompting configuration
│   └── baseline_config.yaml  # Baseline configuration
├── data/                     # Dataset directory
|   ├── processed
├   ├    ├── cluster-dataset-70-25-5.hf
├   ├    ├── safety-dataset-90-5-5.hf
└── results/                   # Output directory
```

Each module and script is designed to be self-contained and reusable. The Hydra configuration system is used to cleanly manage all hyperparameters, model types, and dataset paths.

---

# Setup and Installation

Follow these steps to set up the environment and install all necessary dependencies:

### 1. Clone the Repository

Repository link:
```
https://github.com/Rafaelsarg/bachelor-thesis
```

```bash
git clone <repository-url>
cd bachelor-thesis
```

### 2. Create a Virtual Environment

**Using venv (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

Install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

**Note:** The installation may take several minutes due to the large number of dependencies, particularly PyTorch and CUDA-related packages.

### 4. Download Pre-trained Word2Vec Embeddings (Optional)

Required only for using the `word2vec` vectorizer in baseline models.

- File: `GoogleNews-vectors-negative300.bin.gz` (~1.5 GB)  
- Download from: https://www.kaggle.com/datasets/adarshsng/googlenewsvectors

**Instructions:**
- Extract the file to obtain `GoogleNews-vectors-negative300.bin`
- Move it to the `data/` directory (or adjust the config path accordingly)

### 5. Install and Start Ollama (Optional)

Ollama is required only for running prompting tasks.

**Install Ollama:**
- macOS/Linux:  
```bash
  curl -fsSL https://ollama.ai/install.sh | sh  
```
- Windows:  
  Download the installer from https://ollama.com/

# Running Baseline Experiments

## Overview
The baseline workflow trains and evaluates traditional machine learning models for text classification tasks (e.g., safety or clustering) using vectorizers (Bag-of-Words, TF-IDF, Word2Vec, Sentence-BERT) and classifiers (Logistic Regression, Naive Bayes, SVM, Random Forest). It is configured via `config/baseline_config.yaml` and executed through `scripts/run_baselines.py`, which leverages the preprocessing and pipeline logic in `src/baselines/classification.py`.

## Components
- **Entry Point**: `scripts/run_baselines.py`
  - Main script that loads configurations, preprocesses data, and runs the classification pipeline.
- **Core Logic**: `src/baselines/classification.py`
  - Implements text preprocessing, vectorization, model training with grid search, and result visualization.
- **Configuration File**: `config/baseline_config.yaml`
  - Defines task type, dataset path, vectorizer, classifier, and grid search parameters.

## Running the Script
You can execute baseline experiments in two ways:

1. **Using Command-Line Overrides**:
   - Specify configuration values directly in the command line for quick experiments without modifying the YAML file.
   - Example (basic override):
     ```bash
     python scripts/run_baselines.py task=cluster vectorizer.type=word2vec classifier.type=lr
     ```
   - Example (with hyperparameter overrides):
     ```bash
     python scripts/run_baselines.py \
       task=cluster \
       vectorizer.type=tfidf \
       vectorizer.grid_params.max_features=10000 \
       vectorizer.grid_params.min_df=1 \
       vectorizer.grid_params.ngram_range="[(1,2)]" \
       classifier.type=svm \
       classifier.grid_params.svm.C=1 \
       classifier.grid_params.svm.kernel=linear
     ```
   - **Note**: Command-line overrides are ideal for small changes but can become cumbersome for extensive modifications.

2. **Using the YAML Configuration**:
   - Edit `config/baseline_config.yaml` to specify task, dataset, vectorizer, classifier, and grid search parameters.
   - Run the script without overrides:
     ```bash
     python scripts/run_baselines.py
     ```
   - **Advantage**: Simplifies managing complex configurations, especially for hyperparameter tuning or model changes.

## Outputs
- **Results**: JSON files containing best parameters, label mappings, and evaluation metrics (e.g., F1-macro, classification report) saved in `results_dir`.
- **Visualizations**: Confusion matrix plots saved as PNG files in `results_dir`.

## Configuration Details
For detailed guidance on configuring `baseline_config.yaml`, refer to [Baseline Configuration Explained](#baseline-configuration-explained) in the full project documentation.

# Baseline Configuration Explained

The `config/baseline_config.yaml` file defines settings for running baseline classification experiments with traditional machine learning models and feature extraction methods.

## Task and Dataset
```yaml
task: "cluster"
dataset_map:
  safety: "data/processed/safety-dataset-90-5-5.hf"
  cluster: "data/processed/cluster-dataset-70-25-5.hf"
dataset_path: "${dataset_map[${task}]}"
```
- **`task`**: Task type (`"safety"` for binary classification, `"cluster"` for multi-class).
- **`dataset_map`**: Maps tasks to dataset paths.
- **`dataset_path`**: Resolves to the dataset path based on `task` (e.g., `data/processed/cluster-dataset-70-25-5.hf` for `cluster`).

## Vectorizer Settings
```yaml
vectorizer:
  type: "word2vec"
  grid_params:
    max_features: [5000, 10000]
    min_df: [1, 5]
    ngram_range: [[1, 1], [1, 2]]
```
- **`type`**: Feature extraction method:
  - `"bow"`: Bag-of-Words (token frequency vectors).
  - `"tfidf"`: TF-IDF (scaled token frequency).
  - `"word2vec"`: Pre-trained Word2Vec embeddings.
  - `"sentence_bert"`: Sentence-BERT embeddings.
- **`grid_params`**: Hyperparameters for `"bow"` or `"tfidf"` (ignored for `"word2vec"`, `"sentence_bert"`):
  - `max_features`: Maximum vocabulary size.
  - `min_df`: Minimum document frequency for tokens.
  - `ngram_range`: N-gram ranges (e.g., `[1, 2]` for unigrams and bigrams).

## Classifier Settings
```yaml
classifier:
  type: "lr"
  grid_params:
    lr:
      C: [0.01, 0.1, 1]
      solver: ["liblinear", "saga"]
      max_iter: [300, 1000]
      feature_selection:
        enabled: false
        k: 1000
    nb:
      alpha: [0.1, 1]
    svm:
      C: [0.1, 1]
      kernel: ["linear"]
    rf:
      n_estimators: [100]
      max_depth: [10, 20]
      min_samples_split: [2, 5]
```
- **`type`**: Classifier type:
  - `"lr"`: Logistic Regression.
  - `"nb"`: Naive Bayes.
  - `"svm"`: Support Vector Machine.
  - `"rf"`: Random Forest.
- **`grid_params`**: Hyperparameter grid for the chosen classifier:
  - **lr**:
    - `C`: Inverse regularization strength.
    - `solver`: Optimization algorithm (`"liblinear"`, `"saga"`).
    - `max_iter`: Maximum training iterations.
    - `feature_selection`: Controls feature selection (for `"lr"`, `"svm"` with `"bow"`, `"tfidf"`).
      - `enabled`: Enable/disable feature selection.
      - `k`: Number of top features to select.
  - **nb**:
    - `alpha`: Smoothing parameter.
  - **svm**:
    - `C`: Regularization strength.
    - `kernel`: Kernel type (e.g., `"linear"`).
  - **rf**:
    - `n_estimators`: Number of trees.
    - `max_depth`: Maximum tree depth.
    - `min_samples_split`: Minimum samples to split a node.

## Embedding Paths
```yaml
word2vec_path: "data/GoogleNews-vectors-negative300.bin"
sentence_bert_model: "sentence-transformers/all-MiniLM-L6-v2"
```
- **`word2vec_path`**: Path to pre-trained Word2Vec model (required for `vectorizer.type="word2vec"`).
- **`sentence_bert_model`**: Hugging Face Sentence-BERT model name (required for `vectorizer.type="sentence_bert"`, auto-downloaded on first run).

# Baseline Classification Module

## Overview
The `src/baselines/classification.py` module provides a text classification pipeline for tasks like safety classification or clustering. It supports multiple vectorizers (Bag-of-Words, TF-IDF, Word2Vec, Sentence-BERT) and classifiers (Logistic Regression, Naive Bayes, SVM, Random Forest). The module includes text preprocessing, feature extraction, model training with grid search, and result visualization, configured via `config/baseline_config.yaml` and executed through `scripts/run_baselines.py`.

## Classes

### 1. `Word2VecVectorizer(TransformerMixin)`
- **Purpose**: Converts text into averaged Word2Vec embeddings for classification.
- **Description**: A scikit-learn-compatible transformer that takes a pre-trained Word2Vec model and averages word vectors for each sentence to produce fixed-length embeddings. Returns a numpy array of embeddings.
- **Usage**: Used when `vectorizer_type="word2vec"` to generate embeddings for classifier input.

### 2. `SentenceBERTVectorizer(TransformerMixin)`
- **Purpose**: Encodes text using Sentence-BERT for classification.
- **Description**: A scikit-learn-compatible transformer that uses a pre-trained Sentence-BERT model to encode entire sentences into dense embeddings. Returns a numpy array of embeddings.
- **Usage**: Applied when `vectorizer_type="sentence_bert"` for semantic text representation.

## Functions

### 1. `clean_text_bow_tfidf(text: str) -> str`
- **Purpose**: Preprocesses text for BoW or TF-IDF vectorizers.
- **Description**: Converts text to lowercase, removes digits and punctuation, tokenizes, removes stopwords, and applies lemmatization. Returns cleaned text as a string.
- **Usage**: Used for `vectorizer_type="bow"` or `"tfidf"` to prepare text for traditional feature extraction.

### 2. `clean_text_embeddings(text: str) -> str`
- **Purpose**: Minimally preprocesses text for embedding-based models.
- **Description**: Applies only lowercasing to retain semantic structure suitable for Word2Vec or Sentence-BERT embeddings.
- **Usage**: Used for `vectorizer_type="word2vec"` or `"sentence_bert"`.

### 3. `process_struggle_response_pair(text: str) -> str`
- **Purpose**: Extracts struggle and response from text for safety classification.
- **Description**: Parses text in the format `<struggle> #label# <response> #END#` using regex, returning a combined string or `None` if invalid.
- **Usage**: Applied when `task_type="safety"` to process struggle-response pairs.

### 4. `preprocess_data(dataset_path: str, vectorizer_type: str, task_type: str, verbose: bool = False) -> tuple`
- **Purpose**: Loads and preprocesses a dataset for classification.
- **Description**: Loads a `DatasetDict`, applies appropriate text cleaning based on `vectorizer_type` and `task_type`, and encodes labels as integers. For `task_type="safety"`, converts N/Y labels to Unsafe/Safe. Returns training/test features, labels, and a label mapping dictionary.
- **Usage**: Prepares data for the classification pipeline.

### 5. `get_vectorizer(vectorizer_type: str, params: dict, sentence_bert_model: str = None, word2vec_path: str = None, verbose: bool = False) -> Transformer`
- **Purpose**: Initializes a vectorizer based on the specified type.
- **Description**: Creates a `CountVectorizer` (BoW), `TfidfVectorizer` (TF-IDF), `Word2VecVectorizer`, or `SentenceBERTVectorizer` with provided parameters or model paths.
- **Usage**: Sets up the vectorizer for the pipeline based on configuration.

### 6. `get_classifier(clf_name: str, clf_params: dict, verbose: bool = False, multi_class: bool = False) -> tuple`
- **Purpose**: Initializes a classifier and its hyperparameter grid.
- **Description**: Creates a classifier (Logistic Regression, ComplementNB, SVM, or Random Forest) and defines its grid search parameters. Configures SVM for multi-class if needed.
- **Usage**: Sets up the classifier for the pipeline.

### 7. `save_results(results: dict, results_dir: str, filename: str)`
- **Purpose**: Saves classification results to a JSON file.
- **Description**: Writes a dictionary containing metrics and parameters to a JSON file in `results_dir`.
- **Usage**: Stores evaluation results and best parameters for analysis.

### 8. `run_pipeline(X_train, X_test, y_train, y_test, vectorizer_type: str, vectorizer_params: dict, clf_name: str, clf_params: dict, task_type: str, results_dir: str, feature_selection_cfg: dict = None, word2vec_path: str = None, sentence_bert_model: str = None, verbose: bool = False) -> tuple`
- **Purpose**: Executes the full classification pipeline.
- **Description**: Builds a scikit-learn pipeline with a vectorizer, optional feature selection (for BoW/TF-IDF with Logistic Regression or SVM), and a classifier. Performs grid search, evaluates the model, saves results as JSON, and generates a confusion matrix plot. Returns the best estimator and parameters.
- **Usage**: Core function to train and evaluate models, called by `run_baselines.py`.

## Dependencies
- **Python Libraries**: `re`, `string`, `numpy`, `matplotlib.pyplot`, `seaborn`, `datasets`, `sklearn` (multiple modules), `nltk`, `sentence_transformers`, `gensim`
- **NLTK Resources**: `stopwords`, `punkt`, `wordnet`, `punkt_tab`
- **Configuration**: Global `STOPWORDS` and `LEMMATIZER` for text processing.

## Usage
The module is used within a classification pipeline configured via `config/baseline_config.yaml` and executed by `scripts/run_baselines.py`. It processes a `DatasetDict` with train/test splits, applies text preprocessing, vectorizes data, trains a model, and saves results.

# Running Prompting Experiments

## Overview
The prompting workflow uses local LLMs via Ollama for text classification tasks (e.g., safety or clustering) with zero-shot, few-shot, or custom prompting strategies. It is configured via `config/prompting_config.yaml` and executed through `scripts/run_prompting.py`, which leverages the `OllamaPrompting` class and utilities in `src/prompting/ollama_prompting.py`.

## Components
- **Entry Point**: `scripts/run_prompting.py`
  - Loads configuration, processes test data, sends prompts to the Ollama model, and saves results.
- **Core Logic**: `src/prompting/ollama_prompting.py`
  - Implements the `OllamaPrompting` class for prompt construction and model interaction, plus custom prompting functions.
- **Configuration File**: `config/prompting_config.yaml`
  - Specifies task, prompt type, model, dataset, and prompt template paths.

## Running the Script
Ensure the Ollama server is running locally (`ollama serve`) before executing the script. You can run prompting experiments in two ways:

1. **Using Command-Line Overrides**:
   - Specify configuration values directly for quick experiments without editing the YAML file.
   - Example (basic override):
     ```bash
     python scripts/run_prompting.py task=cluster prompt=zero_shot ollama_model=mistral:7b
     ```
   - Example (with prompt-specific overrides):
     ```bash
     python scripts/run_prompting.py \
       task=safety \
       prompt=few_shot \
       ollama_model=llama3:8b \
       dataset_config.dataset_path=data/processed/safety-dataset-90-5-5.hf
     ```
   - **Note**: Command-line overrides are suitable for small changes but can be unwieldy for complex configurations.

2. **Using the YAML Configuration**:
   - Edit `config/prompting_config.yaml` to set task, prompt type, model, dataset path, and prompt templates.
   - Run the script without overrides:
     ```bash
     python scripts/run_prompting.py
     ```
   - **Advantage**: Simplifies managing detailed settings, especially for few-shot examples or custom prompting.

## Outputs
- **Results**: JSON file with input text, true labels, and predicted labels, saved in `output_directory` (e.g., `<model>_<prompt>.json`).
- **Metrics**: JSON file with evaluation metrics (e.g., F1-score, accuracy), saved as `<model>_<prompt>_metrics.json`.
- **Visualizations**: Confusion matrix plot saved as `<model>_<prompt>_confusion_matrix.png`.

### Configuration Details
For guidance on configuring `prompting_config.yaml`, refer to [Prompting Configuration Explained](#prompting-configuration-explained) in the full project documentation.

# Prompting Configuration Explained

The core of the experiment's setup is managed through the `prompting_config.yaml` file. This file uses Hydra for configuration management, allowing for a flexible and organized way to define parameters.

## User-Editable Parameters

At the top of the `prompting_config.yaml` file, you'll find the primary parameters you can change to control the prompting process:

* **`model`**: Specifies the base LLM to use.
    * **Choices**: `"llama"`, `"mistral"`, `"phi"`
    * **How it works**: This key maps to a specific model identifier in the `ollama_models` section (e.g., `mistral` maps to `mistral:7b`).

* **`task`**: Defines the classification task to be performed.
    * **Choices**: `"safety"`, `"cluster"`
    * **How it works**: This choice determines which dataset to load (from `dataset_config.dataset_map`), which prompt templates to use (from `prompts_path`), and where to save the results (in `output_directory`).

* **`prompt`**: Selects the prompting strategy.
    * **Choices**: `"zero_shot"`, `"few_shot"`, `"custom"`
    * **Note**: The `"custom"` prompt is only available for the `"cluster"` task.
    * **How it works**: This selects the specific prompt structure from the corresponding `.json` file (e.g., `safety_prompts.json`).

## Dynamic Configuration

The rest of the `prompting_config.yaml` file uses variable substitution (`${...}`) to dynamically create paths and select values based on your choices in the user-editable section. For example, if you set `task: safety`, the `dataset_path` automatically resolves to the path specified for `safety` in the `dataset_map`. This system ensures that all components (data, model, prompts, and output directories) are correctly aligned for the selected task and model without needing to manually change paths.

# Prompting Classification Module

The classification logic is primarily handled by two Python scripts: `ollama_prompting.py` and `run_prompting.py`.

### `ollama_prompting.py`: The Core Prompting Engine

This script contains the `OllamaPrompting` class, which is a wrapper for interacting with the Ollama API.

* `__init__(self, model_name, system_instruction)`: Initializes the prompter with the specified Ollama model and a system-level instruction that sets the context for the AI.

* `send_prompt_to_model(...)`: Takes a sequence of messages and sends them to the Ollama model, returning the model's response.

* **Prompt Building Methods**:
    * `build_zero_shot_prompt(...)`: Creates a simple prompt containing only the system instruction and the user's input text.
    * `build_few_shot_prompt(...)`: Constructs a prompt that includes a system instruction, a few examples of input-output pairs, and then the user's input text. This helps the model understand the desired format and task better.
    * `build_custom_prompt(...)`: Used for more complex scenarios. It assembles a prompt with a custom system instruction and contextual examples.
    * `run_custom_prompt(...)`: A specialized function for the `"cluster"` task. It performs a multi-step classification process, narrowing down the possible labels in stages to arrive at a final classification.

### `run_prompting.py`: The Main Execution Script

This script ties everything together using the configuration from `prompting_config.yaml`.

* **Loads Configuration**: It uses Hydra to load the configuration, resolving all the dynamic paths and parameters.
* **Loads Data**: It loads the appropriate test dataset based on the selected `task`.
* **Pulls Model**: It ensures the specified Ollama model is available locally.
* **Loads Prompts**: It reads the corresponding prompt template (`.json` file) for the chosen `task` and `prompt` strategy.
* **Instantiates Prompter**: It creates an instance of the `OllamaPrompting` class.
* **Executes Prompting Loop**: It iterates through each sample in the dataset and:
    * Builds the appropriate prompt (zero-shot, few-shot, or custom).
    * Sends the prompt to the model to get a prediction.
    * Stores the input, actual label, and predicted label.
* **Saves Results & Evaluates**:
    * The results are saved to a `.json` file in the directory specified by the configuration.
    * It calculates and saves performance metrics (like precision, recall, and F1-score) and a confusion matrix image.