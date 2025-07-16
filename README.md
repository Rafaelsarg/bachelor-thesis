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
- [Running Fine-tuning and Inference](#running-fine-tuning-and-inference)
- [Fine-tuning and Inference Configuration Explained](#prompting-configuration-explained)
- [Fine-tuning and Inference Classification Module](#prompting-classification-module)

# Prerequisites

To run the codebase successfully, ensure that the following software and accounts are available and properly configured:

- **Python 3.12.3**  
  All experiments were conducted using Python 3.12.3 on an Ubuntu 24.04 machine. While other Python 3.12+ versions may work, compatibility is not guaranteed.  
  > **Note**: On Ubuntu systems, you must install the development headers with:  
  > `sudo apt install python3.12-dev`  
  It is strongly recommended to use a virtual environment to prevent conflicts with system-wide packages.


- **NVIDIA GPU: A40 or A100 (tested on Linux machines)**  
  Training and fine-tuning large language models is computationally intensive. The code was tested on Linux machines equipped with NVIDIA A40 and A100 40GB or 80GB GPUs. Other CUDA-compatible GPUs may also work, but performance and compatibility are not guaranteed.

- **CUDA-compatible GPU drivers**  
  Ensure that the correct NVIDIA drivers and CUDA toolkit are installed on your system. These are necessary for PyTorch and Hugging Face Transformers to leverage GPU acceleration. **Cuda 12.8 Open** was installed on Ubuntu machine.

- **[Hugging Face](https://huggingface.co) account and access token**  
  Required to download and use pre-trained models from the Hugging Face Hub (e.g., Mistral, Phi, LLaMA).  
  A token will be provided for reproduction purposes. Alternatively, you can [sign up for a free Hugging Face account](https://huggingface.co/join) and [generate your own access token](https://huggingface.co/settings/tokens) with at least `read` permission. Personal huggine face key is provided in the configs in case you do not have one with access to LLama. 

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

# Dataset

The `data/` directory is where all the datasets for this project are stored.

Inside `data/`, you'll find a `processed` subdirectory. This `processed` directory contains datasets that have already been prepared and are in the Hugging Face (`.hf`) format.

Specifically, within `processed`, you'll see:

* **`cluster-dataset-70-25-5.hf`**: This is the processed dataset used for the "cluster" classification task. The numbers "70-25-5" refer to the split percentages for training, testing and validation data, respectively.
* **`safety-dataset-90-5-5.hf`**: This is the processed dataset used for the "safety" classification task. Similarly, "90-5-5" indicates the training, testing and validation data split.

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
- **Results**: JSON files containing best parameters, label mappings, and evaluation metrics (e.g., F1-macro, classification report) saved in `results/baselines/${task}/${classifier.type}/${vectorizer.type}`.
- **Visualizations**: Confusion matrix plots saved as PNG files in `results/baselines/${task}/${classifier.type}/${vectorizer.type}`.

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

#### 1. `clean_text_bow_tfidf(text: str) -> str`
- **Purpose**: Preprocesses text for BoW or TF-IDF vectorizers.
- **Description**: Converts text to lowercase, removes digits and punctuation, tokenizes, removes stopwords, and applies lemmatization. Returns cleaned text as a string.
- **Usage**: Used for `vectorizer_type="bow"` or `"tfidf"` to prepare text for traditional feature extraction.

#### 2. `clean_text_embeddings(text: str) -> str`
- **Purpose**: Minimally preprocesses text for embedding-based models.
- **Description**: Applies only lowercasing to retain semantic structure suitable for Word2Vec or Sentence-BERT embeddings.
- **Usage**: Used for `vectorizer_type="word2vec"` or `"sentence_bert"`.

#### 3. `process_struggle_response_pair(text: str) -> str`
- **Purpose**: Extracts struggle and response from text for safety classification.
- **Description**: Parses text in the format `<struggle> #label# <response> #END#` using regex, returning a combined string or `None` if invalid.
- **Usage**: Applied when `task_type="safety"` to process struggle-response pairs.

#### 4. `preprocess_data(dataset_path: str, vectorizer_type: str, task_type: str, verbose: bool = False) -> tuple`
- **Purpose**: Loads and preprocesses a dataset for classification.
- **Description**: Loads a `DatasetDict`, applies appropriate text cleaning based on `vectorizer_type` and `task_type`, and encodes labels as integers. For `task_type="safety"`, converts N/Y labels to Unsafe/Safe. Returns training/test features, labels, and a label mapping dictionary.
- **Usage**: Prepares data for the classification pipeline.

#### 5. `get_vectorizer(vectorizer_type: str, params: dict, sentence_bert_model: str = None, word2vec_path: str = None, verbose: bool = False) -> Transformer`
- **Purpose**: Initializes a vectorizer based on the specified type.
- **Description**: Creates a `CountVectorizer` (BoW), `TfidfVectorizer` (TF-IDF), `Word2VecVectorizer`, or `SentenceBERTVectorizer` with provided parameters or model paths.
- **Usage**: Sets up the vectorizer for the pipeline based on configuration.

#### 6. `get_classifier(clf_name: str, clf_params: dict, verbose: bool = False, multi_class: bool = False) -> tuple`
- **Purpose**: Initializes a classifier and its hyperparameter grid.
- **Description**: Creates a classifier (Logistic Regression, ComplementNB, SVM, or Random Forest) and defines its grid search parameters. Configures SVM for multi-class if needed.
- **Usage**: Sets up the classifier for the pipeline.

#### 7. `save_results(results: dict, results_dir: str, filename: str)`
- **Purpose**: Saves classification results to a JSON file.
- **Description**: Writes a dictionary containing metrics and parameters to a JSON file in `results_dir`.
- **Usage**: Stores evaluation results and best parameters for analysis.

#### 8. `run_pipeline(X_train, X_test, y_train, y_test, vectorizer_type: str, vectorizer_params: dict, clf_name: str, clf_params: dict, task_type: str, results_dir: str, feature_selection_cfg: dict = None, word2vec_path: str = None, sentence_bert_model: str = None, verbose: bool = False) -> tuple`
- **Purpose**: Executes the full classification pipeline.
- **Description**: Builds a scikit-learn pipeline with a vectorizer, optional feature selection (for BoW/TF-IDF with Logistic Regression or SVM), and a classifier. Performs grid search, evaluates the model, saves results as JSON, and generates a confusion matrix plot. Returns the best estimator and parameters.
- **Usage**: Core function to train and evaluate models, called by `run_baselines.py`.

## Dependencies
- **Python Libraries**: `re`, `string`, `numpy`, `matplotlib.pyplot`, `seaborn`, `datasets`, `sklearn` (multiple modules), `nltk`, `sentence_transformers`, `gensim`
- **NLTK Resources**: `stopwords`, `punkt`, `wordnet`, `punkt_tab`

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
     python scripts/run_prompting.py task=cluster prompt=zero_shot model=mistral
     ```
   - Example (with prompt-specific overrides):
     ```bash
     python scripts/run_prompting.py task=safety prompt=few_shot model=llama 
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

# Running Fine-tuning and Inference

**Fine-Tuning Script**: `scripts/run_finetuning.py`  
**Inference Script**: `scripts/run_inference.py`

These scripts handle fine-tuning and inference for large language models (LLMs) like LLaMA 3, Mistral, or Phi, using QLoRA and Hugging Face PEFT for efficient training. Fine-tuning adapts pre-trained LLMs for dietary topic (`cluster`) or safety classification (`safety`) tasks, while inference generates predictions using the fine-tuned models.

⚠️ **Important**: Fine-tuning for safety classification can take 10–24 hours, while dietary topic classification takes 1–3 hours, depending on parameters. Large batch sizes may cause memory errors due to insufficient VRAM.

## 🧪 Weights & Biases (W&B) Logging

During fine-tuning or inference, you may see the following prompt:

wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results

- **Option 1**: Create a new W&B account (redirects to a browser).
- **Option 2**: Use an existing W&B API key. 
- **Option 3**: Skip logging and run locally.

> ✅ Choose option 3 to disable W&B logging if not needed.

## Fine-Tuning

### User-Editable Parameters

At the top of the `finetune_config.yaml` file, you'll find the primary parameters you can adjust to control the fine-tuning process:

### 🔧 Core Configuration

- **`model`**: Specifies the model family to use.  
  - **Choices**: `"llama"`, `"mistral"`, `"phi"`  
  - **How it works**: Determines which model/tokenizer to load, via the `model_map` (e.g., `"llama"` → `"meta-llama/Meta-Llama-3-8B-Instruct"`).

- **`model_type`**: Defines the type of model head used during fine-tuning.  
  - **Choices**: `"classification"`, `"causal"`  
  - **How it works**:  
    - `"classification"` uses a classification head with cross-entropy loss.  
    - `"causal"` uses a language modeling head with prompt-label token sequences.

- **`task`**: Defines the supervised objective.  
  - **Choices**: `"safety"`, `"cluster"`  
  - **How it works**: Determines which dataset, prompt formatter, and label mappings are used.

- **`model_id`**: Internal model alias for config lookup.  
  - **Examples**: `"llama3"`, `"mistral"`, `"phi"`  
  - **How it works**: Used to link model name and formatter logic from `model_map` and `formatter_registry`.

---

### ⚙️ SFT Configuration (`sft_config`)

- **`batch_size`**: Number of samples per training batch.  
  - **Type**: Integer (e.g., `16`, `32`)

- **`epochs`**: Number of fine-tuning epochs to run.  
  - **Type**: Integer (e.g., `3`, `5`)

- **`learning_rate`**: The learning rate for optimization.  
  - **Type**: Float (e.g., `2e-5`, `5e-5`)  
  - **Note**: Too high may lead to instability; too low may slow down training.

- **`max_seq_length`**: Maximum input sequence length.  
  - **Type**: Integer (e.g., `512`)  
  - **How it works**: Inputs longer than this will be truncated.

- **`loss_function`**: Loss function to use for sequence classification.
  - **Type**: String, options: `custom`, `standart` 
---

### 🧩 LoRA Configuration (`lora_config`)

- **`r`**: Rank of the LoRA decomposition.  
  - **Type**: Integer (e.g., `8`, `16`)

- **`alpha`**: Scaling factor for LoRA updates.  
  - **Type**: Float (e.g., `16.0`, `32.0`)

- **`dropout`**: Dropout probability for LoRA layers.  
  - **Type**: Float (e.g., `0.05`)  
  - **How it works**: Helps prevent overfitting during low-rank adaptation.

- **`target_modules`**: Target modules.  
  - **Type**: default `all-linear`

### 1. **Using Command-Line Overrides**

You can override parameters defined in `config/finetune_config.yaml` directly in the command line. This is useful for quickly testing different model types, learning rates, loss functions, etc.
#### Examples:

```bash
# Fine-tune LLaMA 3 on the cluster classification task using classification head
python scripts/run_finetuning.py model_id=llama3 task=cluster model_type=classification
```

```bash
# Fine-tune Phi-3-mini on safety classification using standard loss and custom learning rate
python scripts/run_finetuning.py model_id=phi task=safety sft_config.loss_function=standard sft_config.lr=1e-5
```

> ✅ **Note:** All other values will still be taken from the YAML file unless explicitly overridden in the command line.

### 2. 🛠 **Using the YAML Configuration**

All parameters can be defined in `config/finetune_config.yaml`, including the task type, model, learning rate, batch size, and LoRA settings.

To run using only the YAML file:

```bash
python scripts/run_finetuning.py
```

This is the preferred way when:
- You are testing multiple configurations
- You want reproducibility
- You need to log and track configurations systematically (e.g., via W&B)

---

### 📁 Results Output Structure

All experiment outputs are saved in:

```
results/finetuning/{model_id}/{task}/{run_name}/
```

The `run_name` is automatically generated using the following format:

```
{model_id}-{task}-{model_type}-lr{learning_rate}-bs{batch_size}-{timestamp}
```

#### Example:
```
llama3-cluster-classification-lr2e-05-bs16-20250715_182741
```

This helps track runs systematically and ensures clear logging when comparing multiple experiments.

## Inference

To perform inference using a fine-tuned model, use the `run_inference.py` script.

You must provide the same model configuration and `run_name` that was used during fine-tuning. This ensures the script loads the correct adapter weights and uses the proper settings for evaluation.

### Required Parameters

- `task`: `"safety"` or `"cluster"`
- `model_id`: `"llama3"`, `"mistral"`, or `"phi"`
- `model_type`: `"classification"` or `"causal"`
- `run_name`: the full name of the run folder created during training

### Example Command

```bash
python scripts/run_inference.py \
    task=cluster \
    model_id=llama3 \
    model_type=classification \
    run_name=llama3-cluster-classification-lr2e-05-bs16-20250715_182741
```

Make sure that the `run_name` exactly matches the folder created during training, which can be found in:

```
results/finetuning/{model_id}/{task}/{run_name}/
```

The script will load the adapter weights and tokenizer from the specified run, and generate predictions on the test set.

Results (predictions and evaluation metrics) will be saved to:
```
results/finetuning/{model_id}/{task}/{run_name}/inference/
```

# Fine-tuning and Inference Classification Module

This module contains all the core components used to fine-tune large language models (LLMs) for classification tasks (e.g., safety or topic/cluster classification) and run inference using the fine-tuned adapters. It supports multiple architectures including LLaMA 3, Mistral, and Phi models, and allows both classification-head fine-tuning and causal LM-style prompt learning via LoRA (Low-Rank Adaptation).

## Fine-tuning

The fine-tuning workflow includes preparing datasets, formatting examples into prompt/label pairs, applying optional LoRA adapters, and training with a Hugging Face-compatible trainer. Models can be configured to run either classification (sequence-level prediction) or causal (next-token prediction) tasks. All logic is modular, enabling consistent handling of diverse model types and datasets.

### Classes

#### 1. `BaseTrainer`
- **Purpose**: Provides shared logic for fine-tuning LLMs.
- **Description**: Abstracts model loading, tokenizer preparation, dataset formatting, and PEFT adapter integration. Handles training configurations and sets up the model for either classification or causal tasks.
- **Usage**: Inherited by `GenericTrainer` to perform task-specific fine-tuning.

#### 2. `GenericTrainer(BaseTrainer)`
- **Purpose**: Fine-tunes LLMs using a provided prompt formatter and configuration.
- **Description**: Implements the `train()` method using Hugging Face's `SFTTrainer`. Prepares tokenized datasets, loads LoRA adapters, and handles metrics/logging. Supports both classification and causal fine-tuning.
- **Usage**: Called in `finetune.py` to launch the training run.

#### 3. `WeightedLossTrainer(SFTTrainer)`
- **Purpose**: Custom Hugging Face trainer that supports weighted loss for imbalanced classification.
- **Description**: Overrides `compute_loss()` to apply class weights in `CrossEntropyLoss`, helping balance the loss when classes (e.g., Safe/Unsafe) are uneven.
- **Usage**: Automatically used when `use_weighted_loss=True` is set in the config.

#### 4. `BasePromptFormatter`
- **Purpose**: Defines the interface for generating prompts and labels.
- **Description**: Abstract base class implemented by all prompt formatters. Requires `format_example()` and `get_label()` methods.
- **Usage**: Inherited by all model-specific formatters (e.g., `LlamaSafetyPromptFormatter`, `MistralTopicPromptFormatter`).

#### 5. `LlamaSafetyPromptFormatter`, `LlamaTopicPromptFormatter`, `MistralSafetyPromptFormatter`, etc.
- **Purpose**: Format input examples for a specific model, task, and type.
- **Description**: Each class extends `BasePromptFormatter` and implements prompt construction and label logic for either `classification` or `causal` tasks, and for `safety` or `cluster`.
- **Usage**: Selected automatically in `finetune.py` or `inference.py` using the `FORMATTER_REGISTRY`.

#### 6. `BaseInference`
- **Purpose**: Provides shared logic for inference on fine-tuned models.
- **Description**: Handles model/tokenizer loading, LoRA adapter application, and batching. Provides utility for tokenizing and decoding model outputs.
- **Usage**: Inherited by `GenericInference`.

#### 7. `GenericInference(BaseInference)`
- **Purpose**: Performs inference on a given dataset using a loaded model and prompt formatter.
- **Description**: Applies formatting to test samples, runs predictions using the correct model head, and returns predicted and reference labels.
- **Usage**: Called in `inference.py` to generate predictions and evaluate results.


### Functions

#### 1. `save_full_config(cfg: DictConfig, output_dir: str)`
- **Purpose**: Saves the full configuration used for training or inference.
- **Description**: Dumps the Hydra config object to a YAML file for reproducibility.
- **Usage**: Called in both `finetune.py` and `inference.py`.

#### 2. `get_tokenized_dataset(tokenizer, dataset: DatasetDict, formatter: BasePromptFormatter, cfg: DictConfig) -> DatasetDict`
- **Purpose**: Prepares tokenized dataset using task-specific formatting.
- **Description**: Applies prompt formatting and tokenization for training or inference. Supports filtering invalid samples.
- **Usage**: Used in `BaseTrainer` and `BaseInference` to prepare data.

#### 3. `load_model_and_tokenizer(cfg: DictConfig)`
- **Purpose**: Loads model and tokenizer using configuration.
- **Description**: Selects the correct base model from `cfg.model_map`, applies quantization if needed, and sets up tokenizer.
- **Usage**: Called inside `BaseTrainer` and `BaseInference`.

#### 4. `get_peft_config(cfg: DictConfig) -> LoraConfig`
- **Purpose**: Returns the LoRA adapter configuration for PEFT.
- **Description**: Builds a `LoraConfig` object with user-specified rank, alpha, dropout, and task type.
- **Usage**: Used in `BaseTrainer` to wrap the model for PEFT.

#### 5. `compute_metrics(eval_pred)`
- **Purpose**: Calculates evaluation metrics from model predictions.
- **Description**: Converts label IDs to strings, computes accuracy, precision, recall, and F1 using `sklearn`.
- **Usage**: Passed to `SFTTrainer` for metric logging.

#### 6. `save_confusion_matrix(references, predictions, labels, output_dir)`
- **Purpose**: Saves a visual confusion matrix plot.
- **Description**: Uses `seaborn` to generate a heatmap and saves it to the `pic/` folder of the output directory.
- **Usage**: Called in `inference.py` after evaluation.

#### 7. `get_prompt_formatter(model_id: str, task: str, model_type: str) -> BasePromptFormatter`
- **Purpose**: Dynamically retrieves the correct formatter class.
- **Description**: Looks up the formatter in the `FORMATTER_REGISTRY` using model ID, task, and model type.
- **Usage**: Called in both `finetune.py` and `inference.py`.