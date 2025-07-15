# Thesis Codebase: Dietary Topic and Safety Classification

This repository contains the codebase developed for thesis experiments involving dietary topic classification and safety assessment of AI-generated nutrition counseling.

---

## 📋 Table of Contents


- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running Baseline Models](#running-baseline-models)
- [Prompting with Ollama](#prompting-with-ollama)
- [Fine-Tuning LLMs](#fine-tuning-llms)
- [Inference](#inference)
- [Output Directory Format](#️output-directory)


---

### Prerequisites

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

### Project Structure

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

### Setup and Installation

Follow these steps to set up the environment and install all necessary dependencies:

#### 1. Clone the Repository

Repository link:
```
https://github.com/Rafaelsarg/bachelor-thesis
```

```bash
git clone <repository-url>
cd bachelor-thesis
```

#### 2. Create a Virtual Environment

**Using venv (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

#### 3. Install Dependencies

Install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

**Note:** The installation may take several minutes due to the large number of dependencies, particularly PyTorch and CUDA-related packages.


#### Task-Specific Setup

The codebase supports three main types of experiments. Follow the setup instructions for the tasks you plan to run:

##### A. Baseline Classification Setup

**Required for:** Traditional ML classification experiments

Download Pre-trained Word2Vec Embeddings

To use the **Word2Vec** feature extraction option in the baseline classifiers, you need to download the pre-trained **Google News Word2Vec** embeddings:

- **File**: `GoogleNews-vectors-negative300.bin.gz` (~1.5GB)
- **Download link**: [GoogleNews-vectors-negative300.bin.gz](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors)

#### Instructions:

1. Download the `.bin` file using the link above.
2. Move the file to data folder


##### B. Fine-tuning Setup

**Required for:** Fine-tuning large language models


**Model Access:**
- **Mistral models:** Available with Hugging Face token
- **Phi models:** Available with Hugging Face token  
- **LLaMA models:** Request access from Meta via [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

**Configuration:**
- Review `config/finetune_config.yaml` for training parameters
- Adjust batch sizes based on your GPU memory
- Set up model paths and dataset configurations

##### C. Prompting Setup

**Required for:** Zero-shot and few-shot prompting experiments

**Install Ollama:**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.com/
```

**Start Ollama service:**
```bash
ollama serve
```

**Configuration:**
- Review `config/prompting_config.yaml` for model and prompt settings


---

### Running Baseline Models

The configuration file defines all parameters needed to run baseline classification experiments using traditional machine learning models and various feature extraction techniques.

#### 🔢 Task and Dataset

```yaml
task: "cluster"
dataset_map:
  safety: "data/processed/safety-dataset-90-5-5.hf"
  cluster: "data/processed/cluster-dataset-70-25-5.hf"
dataset_path: "${dataset_map[${task}]}"
```

- **`task`**: Specifies the task type. Options: `"safety"` or `"cluster"`.
- **`dataset_map`**: Maps each task name to a corresponding dataset path.
- **`dataset_path`**: Dynamically resolves to the correct dataset path using the value of `task`. For example, if `task: cluster`, then `dataset_path` becomes `data/processed/cluster-dataset-70-25-5.hf`.

---

#### 🧠 Vectorizer Settings

```yaml
vectorizer:
  type: "word2vec"
  grid_params:
    max_features: [5000, 10000]
    min_df: [1, 5]
    ngram_range:
      - [1, 1]
      - [1, 2]
```

- **`type`**: Specifies the feature extraction method. Options:
  - `"bow"`: Bag-of-Words — creates a sparse vector based on token frequency.
  - `"tfidf"`: Term Frequency-Inverse Document Frequency — scales token frequency by how unique it is across documents.
  - `"word2vec"`: Pre-trained Word2Vec embeddings — uses dense vectors from a pre-trained model (e.g., Google News vectors).
  - `"sentence_bert"`: Sentence-level transformer embeddings — uses contextualized sentence vectors from transformer models like MiniLM or BERT.

- **`grid_params`**: Hyperparameters for vectorization. Only relevant when using `"bow"` or `"tfidf"`. Ignored for `"word2vec"` and `"sentence_bert"`. Includes:
  - `max_features`: Maximum number of tokens to include in the vocabulary.
  - `min_df`: Minimum number of documents a word must appear in to be included.
  - `ngram_range`: Range of n-grams to consider (e.g., `[1, 2]` includes unigrams and bigrams).

---

#### 🤖 Classifier Settings

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

- **`type`**: Specifies which classifier to use. Options:
  - `"lr"`: Logistic Regression
  - `"nb"`: Naive Bayes
  - `"svm"`: Support Vector Machine
  - `"rf"`: Random Forest

- **`grid_params`**: Hyperparameter search space for each classifier. Only the section matching the chosen `classifier.type` is used in training.

  - **`lr`** (Logistic Regression):
    - `C`: Inverse regularization strength. Smaller values specify stronger regularization.
    - `solver`: Optimization algorithm. Options include:
      - `"liblinear"`: Good for small datasets.
      - `"saga"`: Supports L1 and L2 penalties; scalable.
    - `max_iter`: Maximum number of training iterations.
    - `feature_selection`: Optional subfield to control dimensionality.
      - `enabled`: Boolean flag to enable/disable feature selection.
      - `k`: Number of top features to select if feature selection is enabled.

  - **`nb`** (Naive Bayes):
    - `alpha`: Additive smoothing parameter. Controls the effect of rare features.

  - **`svm`** (Support Vector Machine):
    - `C`: Regularization parameter. Controls trade-off between achieving a low error on training data and minimizing the norm of the weights.
    - `kernel`: Specifies the kernel type to be used. Typically `"linear"` for text data.

  - **`rf`** (Random Forest):
    - `n_estimators`: Number of trees in the forest.
    - `max_depth`: Maximum depth of each tree. Prevents overfitting.
    - `min_samples_split`: Minimum number of samples required to split a node.

---

#### 📦 Embedding and Model Paths

```yaml
word2vec_path: data/GoogleNews-vectors-negative300.bin
sentence_bert_model: sentence-transformers/all-MiniLM-L6-v2
```

- **`word2vec_path`**: Path to the Google News `.bin` Word2Vec file. Required only if `vectorizer.type` is `"word2vec"`.  
  See [Download Pre-trained Word2Vec Embeddings](#a-baseline-classification-setup) for instructions.

- **`sentence_bert_model`**: Name of the Sentence-BERT model hosted on Hugging Face (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).  
  Used only when `vectorizer.type` is set to `"sentence_bert"`. The model will be automatically downloaded and cached by Hugging Face's `transformers` library during the first run.

---

#### 📁 Results Directory

```yaml
results_dir: results/baselines/${task}/${classifier.type}/${vectorizer.type}
```

**Typical contents of the results directory:**

- **`classification_report.json`**  
  Contains detailed evaluation metrics including:
  - Precision
  - Recall
  - F1-score
  - Accuracy  
  Metrics are provided per class, along with macro and weighted averages.

- **`best_params.json`**  
  Stores the best hyperparameters selected by `GridSearchCV` during model training. Useful for reproducibility and future tuning.

- **`confusion_matrix.png`**  
  An optional heatmap visualizing the model's prediction performance across true and predicted classes. Helps in identifying common misclassifications.

---

### ▶️ How to Run Baselines

Baseline experiments can be launched directly from the command line using Hydra configuration overrides.

#### 1. Run with Command-Line Overrides

You can run any baseline experiment without modifying a YAML file by specifying all configuration values inline.

```bash
python scripts/run_baseline.py task=cluster vectorizer.type=word2vec classifier.type=lr
```
or 

```bash
python scripts/run_baseline.py \
  task=cluster \
  vectorizer.type=tfidf \
  vectorizer.grid_params.max_features=10000 \
  vectorizer.grid_params.min_df=1 \
  vectorizer.grid_params.ngram_range="[(1,2)]" \
  classifier.type=svm \
  classifier.grid_params.svm.C=1 \
  classifier.grid_params.svm.kernel=linear \
```

While it's possible to override any configuration directly from the command line, the commands can quickly become long and hard to manage.

---

#### ✅ Recommended: Edit the YAML File Directly

For major updates—such as changing the classifier or vectorizer parameters, or modifying grid search settings—it's much more convenient to open the existing YAML configuration file and update it manually.

Once updated, you can run the script without needing any command-line overrides:

```bash
python scripts/run_baseline.py
```

---

### Prompting with Ollama

This configuration defines how to run zero-shot, few-shot, or custom prompting tasks using locally hosted models via **Ollama**. It supports both the **safety** and **cluster** classification tasks.

```yaml
model: "llama"
task: "cluster"
prompt: "few_shot"

dataset_config:
  dataset_map:
    safety: "data/processed/safety-dataset-90-5-5.hf"
    cluster: "data/processed/cluster-dataset-70-25-5.hf"
  dataset_path: "${dataset_config.dataset_map[${task}]}"

ollama_models:
  llama: "llama3.1:8b"
  mistral: "mistral:7b"
  phi: "phi3:mini"

ollama_model: "${ollama_models[${model}]}"

prompts_path:
  safety: "src/prompting/prompts/safety_prompts.json"
  cluster: "src/prompting/prompts/cluster_prompts.json"

output_directory:
  safety: "results/safety/${prompt}/${model}"
  cluster: "results/cluster/${prompt}/${model}"

label_names:
  safety: ["Unsafe", "Safe"]
  cluster: 
    - "DIET_PLAN_ISSUES"
    - "SOCIAL"
    - "SITUATIONAL"
    - "MOTIVATION"
    - "EMOTIONS"
    - "CRAVING_HABIT"
    - "MENTAL_HEALTH"
    - "ENERGY_EFFORT_CONVENIENCE"
    - "PORTION_CONTROL"
    - "KNOWLEDGE"
    - "HEALTH_CONDITION"
    - "NOT_APPLICABLE"
```

#### 🔧 `model`

Specifies which model to use for prompting. Options:
- `llama`: Meta LLaMA 3.1 8B
- `mistral`: Mistral 7B
- `phi`: Phi-3 Mini

The actual model string is resolved using the `ollama_models` mapping.

#### 🗂 `task`

Defines the classification task to run. Options:
- `safety`: Classifies AI responses as "Safe" or "Unsafe"
- `cluster`: Classifies user struggles into one of 12 dietary topics

This value is used to load the appropriate dataset, prompt file, and output path.

#### 💬 `prompt`

Selects the prompting strategy. Options:
- `zero_shot`
- `few_shot`
- `custom`

Each prompt type changes how the model is guided during inference (e.g., examples included or not).

---

### 📂 Dataset Configuration

The dataset path is selected based on the `task`:

- `safety`: uses `data/processed/safety-dataset-90-5-5.hf`
- `cluster`: uses `data/processed/cluster-dataset-70-25-5.hf`

This is resolved dynamically via the `dataset_map`.

---

### 🧠 Ollama Model Mapping

Maps each model key (e.g., `llama`, `mistral`) to its actual name used by Ollama:

- `llama`: `llama3.1:8b`
- `mistral`: `mistral:7b`
- `phi`: `phi3:mini`

The final model loaded is stored in `ollama_model`.

---

### 📜 Prompt Templates

The path to the prompt definitions (in JSON) is task-specific:

- For `safety`: `src/prompting/prompts/safety_prompts.json`
- For `cluster`: `src/prompting/prompts/cluster_prompts.json`

These files contain the actual input templates used in prompting (zero-shot, few-shot, or custom).

---

### 💾 Output Directory

Results will be saved under a structured folder based on task, prompt type, and model:

- For `safety`: `results/safety/{prompt}/{model}`
- For `cluster`: `results/cluster/{prompt}/{model}`

Each directory contains model predictions, formatted outputs, and optionally confidence scores if available.

---

### 🏷️ Label Names

Used for evaluation and prediction formatting. Dynamically set based on the task:

- `safety`:
  - `["Unsafe", "Safe"]`
- `cluster`:
  - `"DIET_PLAN_ISSUES"`
  - `"SOCIAL"`
  - `"SITUATIONAL"`
  - `"MOTIVATION"`
  - `"EMOTIONS"`
  - `"CRAVING_HABIT"`
  - `"MENTAL_HEALTH"`
  - `"ENERGY_EFFORT_CONVENIENCE"`
  - `"PORTION_CONTROL"`
  - `"KNOWLEDGE"`
  - `"HEALTH_CONDITION"`
  - `"NOT_APPLICABLE"`

These are used for both evaluation and final output labeling.

---

### ▶️ How to Run Prompting

Running a prompting-based experiment is simple and requires minimal setup. The script is designed to work out-of-the-box by selecting the model, task, and prompt type.

#### ✅ Minimal Required Overrides

You only need to change **three parameters**:

- `model`: one of `llama`, `mistral`, or `phi`
- `task`: either `safety` or `cluster`
- `prompt`: one of `zero_shot`, `few_shot`, or `custom`

These three values are enough to load the correct dataset, prompt format, and output folder. All other parameters (like label names, prompt file paths, and Ollama model mappings) are automatically resolved from the configuration and **should not be changed** unless you know what you're doing.

---

#### 🧪 Example Run (with overrides)

You can run a prompting experiment like this:

```bash
python scripts/run_prompting.py model=llama task=cluster prompt=few_shot
```
---

### Fine-Tuning LLMs

This configuration is used for supervised fine-tuning (SFT) of open-source LLMs (like LLaMA, Mistral, and Phi) using either safety or cluster classification tasks.

#### 📋 General Metadata

- `project_name` and `run_name`: Define the experiment's identity and logging format (including timestamp).
- `group` and `tags`: Used for organizing runs in Weights & Biases.
- `task`: Set to either `"safety"` or `"cluster"`.
- `hf_token`: Hugging Face token to access model checkpoints.

#### 🧠 Model Selection

- `model_id`: Selects which model to fine-tune (`llama3`, `mistral`, or `phi`).
- `model_map`: Maps `model_id` to the corresponding Hugging Face model name.
- `model_type`: Either `"classification"` or `"casual"` depending on task format.

#### 📊 Dataset Configuration

- `dataset_path`: Automatically selected based on the task.
- `num_labels`: Set to 2 (safety) or 12 (cluster).
- `label2id` and `label_names`: Task-specific mappings for classification.

---

#### 📁 Output Directory

All fine-tuned model outputs, logs, and checkpoints are saved to the following path:

```
results/finetuning/{model_id}/{task}/{run_name}/
```


The `run_name` is automatically generated based on model and training parameters, using this format:

```
{model_id}-{task}-{model_type}-lr{learning_rate}-bs{batch_size}-{timestamp}
```

For example:
```
llama3-cluster-classification-lr2e-05-bs16-20250706_214753
```

This naming convention uniquely identifies each training run and includes:
- The model used (e.g., `llama3`)
- The task (`safety` or `cluster`)
- The model type (`classification` or `casual`)
- Learning rate (`lr`)
- Batch size (`bs`)
- Timestamp of when the run started

Inside this folder, you will find:
- **Model checkpoints** 
- **Training logs** 
- **Final model tensors**

This structure makes it easy to organize and compare multiple runs.


#### 🧪 SFT Training Settings

- `batch_size`, `epochs`, `lr`, `max_seq_length`: Control training and evaluation.
- `evaluation_strategy`: Evaluates at the end of each epoch.
- `optim`, `scheduler`, `warmup_ratio`: Advanced optimizer settings (e.g., Paged AdamW).
- `loss_function`: Set to `"custom"` to use weighted or task-specific loss.

#### 🔁 LoRA Configuration

- `r`, `alpha`, `dropout`: Control the rank and scale of Low-Rank Adaptation.
- `target_modules`: Modules to which LoRA adapters will be applied (e.g., Q/K/V projections).
- These can be overridden per model if needed.

---

Only `model_id`, `task`, and a few SFT values usually need to be modified. All other parameters are dynamically inferred based on the task.

---

### Inference 

This configuration is used to run inference with a previously fine-tuned model. It loads the adapter weights, applies them to the base model, and generates predictions on the full dataset.

---

#### 🧠 Task and Model Selection

- `task`: Specifies the classification task. Options:
  - `safety`: Binary classification (Safe vs. Unsafe)
  - `cluster`: Multi-class classification (12 dietary topics)
- `model_id`: Which base model to load (`llama3`, `mistral`, or `phi`)
- `model_type`: Should match the training mode, typically `"classification"`

- `run_name`: The name of the fine-tuning run whose adapter weights will be used. It must match the folder name created during fine-tuning, e.g.:

```
llama3-cluster-classification-lr1e-05-bs8-20250705_161928
```

---

#### 🧩 Model & Dataset Mapping

- `model_map`: Maps the `model_id` to its full Hugging Face model name.
- `dataset_map`: Dynamically resolves the dataset path based on `task`.

The final dataset path used for inference is:
