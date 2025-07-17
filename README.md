# Thesis Codebase: Dietary Topic and Safety Classification

This repository contains the codebase developed for thesis experiments involving dietary topic classification and safety assessment of AI-generated nutrition counseling.

## 📋 Table of Contents

- [📁 Project Structure](#project-structure)  
- [📦 Prerequisites](#prerequisites)  
- [🗂️ Dataset Overview](#dataset-overview)  
- [⚙️ Running Baseline Experiments](#running-baseline-experiments)  
- [🔧 Baseline Configuration Explained](#baseline-configuration-explained)  
- [🧠 Baseline Classification Module](#baseline-classification-module)  
- [💬 Running Prompting Experiments](#running-prompting-experiments)  
- [🛠️ Prompting Configuration Explained](#prompting-configuration-explained)  
- [🧠 Prompting Classification Module](#prompting-classification-module)  
- [🏋️ Running Fine-Tuning and Inference](#running-fine-tuning-and-inference)  
- [🧠 Fine-tuning & Inference Classification Module](#fine-tuning--inference-classification-module)  


## Project Structure

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


## 📦 Prerequisites

Before running any experiments or scripts in this repository, make sure your environment is properly set up. Below is a list of required tools, hardware, and accounts to ensure full compatibility and reproducibility.

---

### ✅ Required Software and Tools

- **Python 3.12.3**  
  All experiments were conducted using Python 3.12.3 on Ubuntu 24.04. While other Python 3.12+ versions may work, compatibility is not guaranteed.  

  > 💡 **Note:** On Ubuntu, install the development headers with:  
  > `sudo apt install python3.12-dev`  

  It is strongly recommended to use a virtual environment to avoid conflicts with system-wide packages.

- **CUDA-compatible NVIDIA GPU**  
  Fine-tuning large models is computationally intensive. The code was tested on:
  - NVIDIA A40 (40GB)
  - NVIDIA A100 (40GB and 80GB)  

  Other CUDA-capable GPUs may work, but performance and compatibility are not guaranteed.

- **CUDA Drivers and Toolkit**  
  Ensure you have the correct NVIDIA drivers and CUDA Toolkit installed. These are required for GPU acceleration via PyTorch and Hugging Face Transformers.  

  The tested version on Ubuntu was:  
  `CUDA 12.8 Open`

---

### 🔐 Required Accounts and API Keys

- **[Hugging Face](https://huggingface.co) Account**  
  Required to download pre-trained models (e.g., Mistral, Phi, LLaMA) from the Hugging Face Hub.  

  You must:
  - [Sign up for a free account](https://huggingface.co/join)
  - [Generate your own access token](https://huggingface.co/settings/tokens) with at least `read` permission

  ✅ **Token Placement**  
  Insert your Hugging Face token as a string in the following configuration files:

  ```yaml
  # config/finetune_config.yaml
  hf_token: "your_token_here"

  # config/inference_config.yaml
  hf_token: "your_token_here"
  ```

  ⚠️ **LLaMA Access Required**  
  For models like `meta-llama/Meta-Llama-3-8B-Instruct`, you must request access from Meta via the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) before using the model.

- **[Weights & Biases (wandb)](https://wandb.ai/) Account**  
  Used for experiment tracking and visualization (loss, accuracy, F1, configuration, etc.).

  Steps:
  - [Sign up here](https://wandb.ai/)
  - [Generate your API key](https://wandb.ai/authorize)

  💡 **Tip:** Educational account is available, [request free academic access](https://wandb.ai/site/education).

  You can:
  - Use your own account and key
  - Or disable W&B logging

  ⚠️ **No W&B? No problem!**  
  During the run, you’ll be asked whether to enable logging. 

  ```bash
  wandb: (1) Create a W&B account
  wandb: (2) Use an existing W&B account
  wandb: (3) Don't visualize my results
  ```

  - **Option 1**: Create a new W&B account (redirects to a browser).
  - **Option 2**: Use an existing W&B API key. 
  - **Option 3**: Skip logging and run locally.

> ✅ Choose option 3 to disable W&B logging if not needed.


## Dataset Overview

All datasets used in this project are stored in the `data/` directory. These datasets have been preprocessed and saved in Hugging Face’s optimized `.hf` format for efficient loading and training.

---

### 📁 Directory Structure

The key subdirectory is:

```
data/
└── processed/
├──────|cluster-dataset-70-25-5.hf
└──────|safety-dataset-90-5-5.hf
```
---
### 📊 Available Datasets

- **`cluster-dataset-70-25-5.hf`**  
  This dataset is used for the **cluster classification task**.  
  - Format: Hugging Face `.hf`
  - Split:  
    - **70%** training  
    - **25%** testing  
    - **5%** validation  

- **`safety-dataset-90-5-5.hf`**  
  This dataset is used for the **safety classification task**.  
  - Format: Hugging Face `.hf`  
  - Split:  
    - **90%** training  
    - **5%** testing  
    - **5%** validation  
---

##  Setup and Installation

Follow these steps to set up your environment and install all necessary dependencies for running experiments.

---

### 1. Clone the Repository

Clone the project from GitHub:

```bash
git clone https://github.com/Rafaelsarg/bachelor-thesis
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
- Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?pli=1&resourcekey=0-wjGZdNAUop6WykTtMip30g
- The link is accessed 16/07/2025

**Instructions:**
- Extract the file to obtain `GoogleNews-vectors-negative300.bin`
- Move it to the `data/` directory (or adjust the config path accordingly)

### 5. Install and Start Ollama (Optional)

Ollama is required only for running prompting tasks.

**Install Ollama:**
- macOS/Linux:  
```bash
  curl -fsSL https://ollama.com/install.sh | sh 
```
- Windows:  
  Download the installer from https://ollama.com/

##  Running Baseline Experiments

This module trains and evaluates traditional machine learning models for text classification tasks—such as safety classification or topic clustering—using various vectorization methods and classifiers.

---

### 📌 Overview

Baseline experiments use combinations of:

- **Vectorizers**: Bag-of-Words, TF-IDF, Word2Vec, Sentence-BERT  
- **Classifiers**: Logistic Regression, Naive Bayes, SVM, Random Forest

The entire pipeline is configured via `config/baseline_config.yaml` and executed with the script `scripts/run_baselines.py`. The core training and evaluation logic is implemented in `src/baselines/classification.py`.

---

### 🧱 Key Components

- **📄 Entry Point** – `scripts/run_baselines.py`  
  Loads configuration, performs preprocessing, and executes the full classification pipeline.

- **⚙️ Core Logic** – `src/baselines/classification.py`  
  Handles vectorization, model training (with GridSearchCV), and result reporting/visualization.

- **🧼 Preprocessing** – `src/baselines/data_processor.py`  
  Performs text cleaning and transformation prior to vectorization.

- **⚙️ Configuration File** – `config/baseline_config.yaml`  
  Defines:
  - Task type (`safety` or `cluster`)
  - Dataset path
  - Vectorizer and its parameters
  - Classifier and its hyperparameters

---

### ▶️ How to Run

You can run baseline experiments in two ways:

#### ✅ Option 1: With Command-Line Overrides

Useful for quick experiments without editing the YAML file.

You can mix and match different **tasks**, **vectorizers** and **classifiers** using Hydra-style key-value overrides.

---


### 🧠 Valid Task Options:
- `safety` – Safety Classification
- `cluster` – Cluster Classification

### 🧠 Valid Vectorizer Options:
- `bow` – Bag-of-Words
- `tfidf` – Term Frequency–Inverse Document Frequency
- `word2vec` – Pretrained Word2Vec embeddings
- `sentence_bert` – Sentence-level embeddings via pretrained SBERT

### 🤖 Valid Classifier Options:
- `lr` – Logistic Regression
- `nb` – Complement Naive Bayes
- `svm` – Support Vector Machine
- `rf` – Random Forest

---

**Basic examples:**
```bash
python scripts/run_baselines.py task=cluster vectorizer.type=bow classifier.type=lr
```

```bash
python scripts/run_baselines.py task=safety vectorizer.type=bow classifier.type=nb
```

```bash
python scripts/run_baselines.py task=cluster vectorizer.type=sentence_bert classifier.type=svm
```

**With hyperparameter overrides:**
```bash
python scripts/run_baselines.py task=cluster \
  vectorizer.type=tfidf \
  vectorizer.grid_params.max_features=[10000] \
  vectorizer.grid_params.min_df=[1] \
  vectorizer.grid_params.ngram_range=[[1,1]] \
  classifier.type=svm \
  classifier.grid_params.svm.C=[1] \
```

#### ✅ Option 2: Using the YAML Configuration File

This method is ideal for reproducible and maintainable experiments. All model and training parameters are defined in a single configuration file.

1. **Open the file** `config/baseline_config.yaml`
2. **Edit the following fields as needed:**
   - `task`: `"safety"` or `"cluster"`
   - `vectorizer`: Choose from `"bow"`, `"tfidf"`, `"word2vec"`, or `"sentence_bert"`
   - `classifier`: Choose from `"lr"`, `"nb"`, `"svm"`, or `"rf"`
   - Define any grid search parameters under the corresponding section

3. **Run the script with no overrides:**

```bash
python scripts/run_baselines.py
```

### 📁 Outputs
- **Results**: JSON files containing best parameters, label mappings, and evaluation metrics (e.g., F1-macro, classification report) saved in `results/baselines/${task}/${classifier.type}/${vectorizer.type}`.
- **Visualizations**: Confusion matrix plots saved as PNG files in `results/baselines/${task}/${classifier.type}/${vectorizer.type}`.

## Baseline Configuration Overview

Baseline experiments are configured via `config/baseline_config.yaml`. Below is a concise overview of key sections.

---

### 🧪 Task and Dataset

```yaml
task: "cluster"
dataset_map:
  safety: "data/processed/safety-dataset-90-5-5.hf"
  cluster: "data/processed/cluster-dataset-70-25-5.hf"
dataset_path: "${dataset_map[${task}]}"
```
- **`task`**: Task type (`"safety"` or `"cluster"` for dietary topic classification).
- **`dataset_map`**: Maps tasks to dataset paths.
- **`dataset_path`**: Resolves to the dataset path based on `task` (e.g., `data/processed/cluster-dataset-70-25-5.hf` for `cluster`).

### 🔤 Vectorizer Settings
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

### 🤖 Classifier Settings
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

### Embedding Paths
```yaml
word2vec_path: "data/GoogleNews-vectors-negative300.bin"
sentence_bert_model: "sentence-transformers/all-MiniLM-L6-v2"
```
- **`word2vec_path`**: Path to pre-trained Word2Vec model (required for `vectorizer.type="word2vec"`).
- **`sentence_bert_model`**: Hugging Face Sentence-BERT model name (required for `vectorizer.type="sentence_bert"`, auto-downloaded on first run).

## Baseline Classification Module

This module (`src/baselines/classification.py`) defines the classification pipeline used for both **safety** and **topic (cluster)** classification. It supports multiple vectorizers and classifiers, includes preprocessing logic, and integrates with `config/baseline_config.yaml` and `scripts/run_baselines.py`.

---

#### `Word2VecVectorizer(TransformerMixin)`
- Averages pretrained Word2Vec word embeddings into a fixed-length vector.

#### `SentenceBERTVectorizer(TransformerMixin)`
- Encodes full sentences using a pretrained Sentence-BERT model.

---

### 🧼 Preprocessing Functions

| Function | Purpose |
|---------|---------|
| `clean_text_bow_tfidf(text)` | Full text cleaning (for BoW/TF-IDF) |
| `clean_text_embeddings(text)` | Minimal cleaning (for Word2Vec/Sentence-BERT) |
| `process_struggle_response_pair(text)` | Extracts `<struggle>#label#<response>` format for safety tasks |
| `preprocess_data(...)` | Loads dataset, cleans text, encodes labels |

---

### 🔧 Utility Functions

| Function | Description |
|----------|-------------|
| `get_vectorizer(...)` | Returns vectorizer based on type and config |
| `get_classifier(...)` | Returns classifier and its hyperparameter grid |
| `save_results(...)` | Saves metrics and config as JSON |
| `run_pipeline(...)` | Runs the full classification pipeline: vectorization → optional feature selection → classification → evaluation and visualization |

---

### 📁 Example Pipeline Flow
1. Clean and load dataset via `preprocess_data`
2. Create vectorizer and classifier objects
3. Fit pipeline using `run_pipeline`
4. Save metrics and confusion matrix to `results/`

---

### 📚 Dependencies

- **Python libs**: `re`, `string`, `numpy`, `matplotlib`, `seaborn`, `datasets`, `sklearn`, `nltk`, `sentence_transformers`, `gensim`
- **NLTK corpora**: `punkt`, `stopwords`, `wordnet`

## Running Prompting Experiments

This module enables zero-shot, few-shot, or custom prompt-based classification using local LLMs via **Ollama**. It supports both safety and topic (cluster) classification tasks.

---

### 📌 Overview

Prompting is configured through `config/prompting_config.yaml` and executed via:

```bash
python scripts/run_prompting.py
```

### 🔧 Key Components

- **🟢 Entry Point** – `scripts/run_prompting.py`  
  Loads configuration, processes test data, sends prompts to the Ollama model, and saves the outputs.

- **🧠 Core Logic** – `src/prompting/ollama_prompting.py`  
  Contains the `OllamaPrompting` class responsible for:
  - Prompt construction based on the selected strategy (`zero_shot`, `few_shot`, `custom`)
  - Interfacing with the local Ollama model
  - Returning model predictions

- **⚙️ Configuration File** – `config/prompting_config.yaml`  
  Defines:
  - `task`: `safety` or `cluster`
  - `prompt`: Prompting strategy (`zero_shot`, `few_shot`, or `custom`)
  - `model`: LLM identifier (e.g., `llama`, `mistral`, `phi`)

---

### ▶️ How to Run

> ⚠️ Make sure the Ollama server is running locally before executing:
> ```bash
> ollama serve
> ```

#### ✅ Option 1: Command-Line Overrides

Useful for quick testing without editing the YAML file.

**Example – Zero-shot Cluster Classification**
```bash
python scripts/run_prompting.py task=cluster prompt=zero_shot model=mistral
```

**Example – Few-shot Safety Classification**
```bash
python scripts/run_prompting.py task=safety prompt=few_shot model=llama
```

**Example – Custom Cluster Classification**
```bash
python scripts/run_prompting.py task=cluster prompt=custom model=phi
```

⚠️ Command-line overrides are ideal for small experiments. For reproducibility and complex prompts, use the YAML config instead.

---

#### ✅ Option 2: YAML Configuration

This method is recommended for larger experiments, prompt template management, or when sharing setups with others.

1. Open `config/prompting_config.yaml`
2. Specify the following fields:
   - `task`: `"safety"` or `"cluster"`
   - `prompt`: `"zero_shot"`, `"few_shot"`, or `"custom"`
   - `model`: e.g., `"llama"`, `"mistral"`, `"phi"`

3. Run the script without overrides:

```bash
python scripts/run_prompting.py
```

### 📤 Outputs

All outputs are saved under:
```
results/prompting/{task}/{prompt}/{model}/
```

This directory contains the following files:

- All Input-Actual-Predicted triples
- File with all the metrics
- Confusion matrix png

## Prompting Classification Module

The logic for prompt-based classification is split across two main scripts:

---

### 📄 `src/prompting/ollama_prompting.py` – Core Prompting Engine

This file defines the `OllamaPrompting` class, which interfaces with the local Ollama API and handles prompt construction for different strategies.

#### 🔧 Key Methods

- `__init__(self, model_name, system_instruction)`  
  Initializes the prompting engine with a selected model and system-level context.

- `send_prompt_to_model(messages)`  
  Sends a list of messages (in chat format) to the Ollama model and returns the response.

#### 🧱 Prompt Construction

- `build_zero_shot_prompt(...)`  
  Constructs a minimal prompt with just system instruction and user input.

- `build_few_shot_prompt(...)`  
  Adds a few input-output examples before the test input to guide the model.

- `build_custom_prompt(...)`  
  Builds a fully custom prompt with examples, metadata, or reasoning chains.

- `run_custom_prompt(...)`  
  Used for the `cluster` task — performs a multi-step narrowing-down classification across multiple definition groups to arrive at the final label.

---

### 🚀 `scripts/run_prompting.py` – Execution Script

This script runs the prompting pipeline using the configuration from `config/prompting_config.yaml`.

#### 🔄 Workflow Steps

- **🔧 Load Config**  
  Uses Hydra to load `prompting_config.yaml`, resolving task, prompt type, model name, etc.

- **📥 Load Dataset**  
  Loads the appropriate test dataset based on the selected `task`.

- **⬇️ Pull Model**  
  Checks and pulls the specified Ollama model if not already present locally.

- **📜 Load Prompt Template**  
  Loads `.json` prompt templates for `few_shot` and `custom` prompts.

- **🤖 Initialize Prompter**  
  Creates an instance of `OllamaPrompting` with the configured model and system message.

- **🔁 Prompting Loop**  
  Iterates through each test sample and:
  - Builds a prompt (zero-shot, few-shot, or custom)
  - Sends it to the model and receives a prediction
  - Collects the original text, true label, and predicted label

- **💾 Save & Evaluate**  
  - Saves results to `results/prompting/{task}/{prompt}/{model}/results.json`
  - Computes metrics (accuracy, precision, recall, F1) and saves them as a `.json`
  - Generates and saves a confusion matrix as a `.png` image

---

This modular design makes it easy to plug in new models, modify prompting strategies, or extend to other tasks.

## 🏋️ Running Fine-tuning and Inference

**Fine-Tuning Script**: `scripts/run_finetuning.py`  
**Inference Script**: `scripts/run_inference.py`

These scripts allow you to fine-tune and evaluate large language models (LLaMA 3, Mistral, or Phi) for:

- **Dietary Topic Classification** (`cluster`)
- **Safety Classification** (`safety`)

Fine-tuning is performed using **QLoRA** + **Hugging Face PEFT**, enabling efficient training on consumer-grade GPUs.

> ⚠️ **Note**:  
> - **Safety fine-tuning** may take **10–24 hours**  
> - **Cluster fine-tuning** typically takes **1–3 hours**  
> - Large batch sizes may cause out-of-memory (OOM) errors depending on your VRAM.

---

## 🧪 Weights & Biases (W&B) Logging

During training or inference, you may see:

```text
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```

(1) Opens browser to register a new account
(2) Use your existing API key
(3) Disables logging (local run only)

✅ Select option 3 if you don’t want to use W&B

## 🔧 Fine-Tuning Configuration

All fine-tuning settings are defined in `config/finetune_config.yaml`. Below are the key configuration blocks:

---

### 🔹 Core Settings

```yaml
model: "llama"
model_type: "classification"
task: "cluster"
model_id: "llama3"
```

| Parameter     | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `model`       | Base model family. Options: `"llama"`, `"mistral"`, `"phi"`                |
| `model_type`  | Type of training head. `"classification"` adds a classifier head, `"causal"` uses next-token prediction |
| `task`        | Task to fine-tune on. Choices: `"safety"` (binary) or `"cluster"` (multi-class) |
| `model_id`    | Internal alias used for model loading, checkpoint saving, and prompt formatting (e.g., `"llama3"`, `"mistral"`) |

---

### ⚙️ `sft_config` – Supervised Fine-Tuning Parameters

| Parameter                    | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `batch_size`                | Number of samples per training batch (e.g., `6`, `16`)                      |
| `epochs`                    | Number of complete passes through the training set                         |
| `lr`                        | Learning rate for the optimizer (e.g., `2e-5`, `5e-5`)                      |
| `evaluation_strategy`       | When to run evaluation. Options: `"epoch"` or `"steps"`                    |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients before a backward pass            |
| `max_grad_norm`             | Maximum gradient norm for clipping (helps with training stability)         |
| `lr_scheduler_type`         | Scheduler for adjusting learning rate. Common choice: `"cosine"`           |
| `warmup_ratio`              | Fraction of total training steps to warm up the learning rate              |
| `optim`                     | Optimizer to use. `"paged_adamw_32bit"` is efficient for QLoRA             |
| `logging_steps`             | Frequency (in steps) of logging metrics during training                    |
| `max_seq_length`            | Maximum number of tokens per input. Longer sequences will be truncated     |
| `weight_decay`              | L2 regularization coefficient applied to model weights                     |
| `loss_function`             | Loss function for classification tasks: `"standard"` or `"custom"`         |

> 🔹 `loss_function` is only used when `model_type=classification`. It is ignored for `causal` models.

---

### 🧩 `lora_config` – LoRA Adaptation Settings

| Parameter         | Description                                                              |
|------------------|--------------------------------------------------------------------------|
| `r`               | Rank of the low-rank adapter matrices (e.g., 8, 16)                     |
| `alpha`           | Scaling factor applied to the LoRA weights                              |
| `dropout`         | Dropout applied within the LoRA layers to prevent overfitting           |
| `target_modules`  | Specifies which layers to apply LoRA to (typically `"all-linear"`)      |


## ▶️ How to Run Fine-Tuning

You can launch fine-tuning in two ways: using command-line overrides or relying entirely on the YAML config file.

---

### 1️⃣ Command-Line Overrides

Override any config value directly via CLI. This is useful for quick tests or one-off experiments.

#### 🔹 Basic Examples


You can override parameters defined in `config/finetune_config.yaml` directly in the command line. This is useful for quickly testing different model types, learning rates, loss functions, etc.

Fine-tune LLaMA 3 on the cluster classification task using classification head
```bash
python scripts/run_finetuning.py model_id=llama task=cluster model_type=classification
```

Fine-tune Mistral-7B on safety classification using standard loss and custom learning rate
```bash
python scripts/run_finetuning.py model_id=mistral task=safety  model_type=classification sft_config.loss_function=standard sft_config.lr=1e-5
```

Fine-tune Mistral-7B on safety classification using custom loss
```bash
python scripts/run_finetuning.py model_id=llama task=cluster  model_type=classification sft_config.loss_function=custom 
```

Fine-tune Mistral-7B on safety classification with causal language modelling
```bash
python scripts/run_finetuning.py model_id=llama task=cluster  model_type=causal 
```


#### ⚠️ Notes:
- The `loss_function` is **only applied** when `model_type=classification`.  
  It is **ignored** for `model_type=causal`.
- `model_id=phi` **does not support** `model_type=classification` and must be used with `model_type=causal`.

> **Note:** All other values will still be taken from the YAML file unless explicitly overridden in the command line.


⚠️ loss_function is only applied if model_type=classification.
model_id=phi supports only model_type=causal.

✅ Values not overridden on the command line will still be loaded from config/finetune_config.yaml.

### 2️⃣ YAML-Only Configuration (Recommended)

You can run the fine-tuning process using only the YAML file by specifying all parameters in `config/finetune_config.yaml`.

This is the preferred method when:
- You want full reproducibility
- You're testing multiple configurations

Then run without overriding:

```bash
python scripts/run_finetuning.py 
```

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


## 🔍 Inference

You must specify the same parameters that were used during training to ensure compatibility with the saved model.

---

### 📌 Required Parameters

| Parameter        | Description                                                                          |
|------------------|--------------------------------------------------------------------------------------|
| `task`           | Task used during fine-tuning: `"safety"` or `"cluster"`                             |
| `model_id`       | Internal model identifier: `"llama"`, `"mistral"`, or `"phi"`                       |
| `model_type`     | Training head used: `"classification"` or `"causal"`                                |
| `sequence_length`| (Optional) Maximum sequence length (defaults to `512`)                               |
| `run_name`       | Name of the fine-tuning run (auto-generated and used as folder name)                |

> ⚠️ All parameters **must match** those used during fine-tuning.  
> Mismatches will lead to errors or incorrect results.

---

### ✅ Example Command

```bash
python scripts/run_inference.py \
    task=cluster \
    model_id=llama \
    model_type=classification \
    run_name=llama3-cluster-classification-lr2e-05-bs16-20250715_182741
```

Make sure the `run_name` exactly matches the folder created during fine-tuning. You can find it under:

```text
results/finetuning/{model_id}/{task}/{run_name}/
```


The script will:
- Load the adapter weights (trained LoRA layers)
- Load the tokenizer associated with the base model
- Prepare the test dataset for the specified `task`
- Generate predictions
- Compute evaluation metrics

---

### 📤 Output Directory

All inference outputs are saved to the same directory of model run where adapter weights are saved:

```text
results/finetuning/{model_id}/{task}/{run_name}/
```

This directory includes:

| File                    | Description                                                   |
|-------------------------|---------------------------------------------------------------|
| `predictions.json`      | JSON list of inputs, true labels, and model predictions       |
| `results.json`          | Evaluation metrics: accuracy, precision, recall, F1-score     |
| `images/confusion_matrix.png`  | PNG image showing confusion matrix of predicted vs. true labels |

These outputs allow you to inspect model performance, generate reports, or compare runs across models and configurations.

## Fine-tuning & Inference Classification Module

This module contains the core components used to fine-tune large language models (LLMs) for classification tasks (e.g., `safety` or `cluster`) and perform inference using the fine-tuned adapters.

It supports:
- Multiple model families: **LLaMA 3**, **Mistral**, **Phi**
- Two training modes:
  - **Classification head fine-tuning** (`model_type: classification`)
  - **Causal LM-style prompt learning** (`model_type: causal`)

Fine-tuning uses **LoRA (Low-Rank Adaptation)** via Hugging Face PEFT for efficient training.

---

## 🏋️ Fine-tuning Workflow

The fine-tuning process includes:
- Dataset formatting using model/task-specific prompt formatters
- LoRA adapter configuration
- Hugging Face-compatible training
- Optional weighted loss for class imbalance

---

### 📦 Classes

#### 1. `BaseTrainer`
- Sets up model, tokenizer, dataset formatting, and LoRA integration.
- Base class for reusable fine-tuning logic.

#### 2. `GenericTrainer(BaseTrainer)`
- Implements `train()` using Hugging Face `SFTTrainer`.
- Handles task-specific prompt formatting and LoRA injection.
- Used in `finetune.py`.

#### 3. `WeightedLossTrainer(SFTTrainer)`
- Custom `Trainer` that supports class-weighted loss via `CrossEntropyLoss`.
- Applied when `use_weighted_loss=True`.

#### 4. `BasePromptFormatter`
- Abstract class defining `format_example()` and `get_label()` interface.
- Extended by all task/model-specific formatters.

#### 5. Model-Specific Prompt Formatters

Each model (`mistral`, `llama3`, `phi`) has dedicated prompt formatters for every task (`safety`, `cluster`) and model type (`causal`, `classification`). These formatters inherit from `BasePromptFormatter` and define how examples are transformed into prompt-label pairs.

They are selected dynamically from the `FORMATTER_REGISTRY` based on:
- `model_id` (e.g., `"llama3"`)
- `task` (e.g., `"safety"`)
- `model_type` (e.g., `"classification"`)

#### 🔁 Registered Formatters:

| Model     | Task     | Model Type       | Formatter Class                              |
|-----------|----------|------------------|-----------------------------------------------|
| `mistral` | `safety` | `causal`         | `MistralSafetyPromptFormatterCausal`         |
|           |          | `classification` | `MistralSafetyPromptFormatterClassification` |
|           | `cluster`| `causal`         | `MistralTopicPromptFormatterCausal`          |
|           |          | `classification` | `MistralTopicPromptFormatterClassification`  |
| `llama3`  | `safety` | `causal`         | `LlamaSafetyPromptFormatterCausal`           |
|           |          | `classification` | `LlamaSafetyPromptFormatterClassification`   |
|           | `cluster`| `causal`         | `LlamaTopicPromptFormatterCausal`            |
|           |          | `classification` | `LlamaTopicPromptFormatterClassification`    |
| `phi`     | `safety` | `causal`         | `PhiSafetyPromptFormatterCausal`             |
|           |          | `classification` | `PhiSafetyPromptFormatterClassification`     |
|           | `cluster`| `causal`         | `PhiTopicPromptFormatterCausal`              |
|           |          | `classification` | `PhiTopicPromptFormatterClassification`      |


#### ⚠️ Notes: 
- Phi Causal is not used!!!!!!

#### 6. `BaseInference`
- Handles loading models, LoRA adapters, and batching logic for evaluation.
- Base class for reusable inference logic.

#### 7. `GenericInference(BaseInference)`
- Performs full evaluation using tokenized test data.
- Applies formatting, runs inference, and returns predictions/labels.
- Called by `inference.py`.

---

## 🛠️ Key Functions

#### 1. `save_full_config(cfg: DictConfig, output_dir: str)`
- Saves complete config as `.yaml` for reproducibility.

#### 2. `get_tokenized_dataset(...) -> DatasetDict`
- Tokenizes and formats a Hugging Face `DatasetDict` using the selected formatter.
- Filters out invalid examples if needed.

#### 3. `load_model_and_tokenizer(cfg: DictConfig)`
- Loads the base model and tokenizer from `cfg.model_map`.
- Applies quantization and prepares the tokenizer.

#### 4. `get_peft_config(cfg: DictConfig) -> LoraConfig`
- Returns a configured LoRA adapter object for PEFT integration.

#### 5. `compute_metrics(eval_pred)`
- Computes accuracy, precision, recall, and F1 using `sklearn.metrics`.

#### 6. `save_confusion_matrix(...)`
- Plots and saves a confusion matrix as a PNG using `seaborn`.

#### 7. `get_prompt_formatter(model_id, task, model_type) -> BasePromptFormatter`
- Dynamically selects the correct prompt formatter from `FORMATTER_REGISTRY`.

---

This modular structure ensures a clean separation between model logic, formatting strategy, and training/inference mechanics. It makes it easy to extend the pipeline to new tasks or LLM architectures.
