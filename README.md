# Thesis Codebase: Dietary Topic and Safety Classification

This repository contains the codebase developed for thesis experiments involving dietary topic classification and safety assessment of AI-generated nutrition counseling.

---

## 📋 Table of Contents


- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Main Scripts](#running-the-main-scripts)
- [Running Baseline Models](#running-baseline-models)
- [Prompting with Ollama](#prompting-with-ollama)
- [Fine-Tuning LLMs](#fine-tuning-llms)
- [Inference](#inference)
- [Output Directory Format](#️output-directory)


---

## Prerequisites

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

---

## Setup and Installation

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

---

## Running the Main Scripts

This project includes three main workflows: running traditional ML baselines, prompting local LLMs via Ollama, and fine-tuning models with PEFT followed by inference.


### 1. Baselines

**Entry Point**: `scripts/run_baseline.py`  
**Main Source Code**: `src/baselines/classification.py`

This script runs traditional ML pipelines using vectorizers like BoW, TF-IDF, Word2Vec, or Sentence-BERT with classifiers such as Logistic Regression, Naive Bayes, SVM, or Random Forest.


You can run any baseline experiment without modifying a YAML file by specifying all configuration values in command line.

```bash
python scripts/run_baseline.py task=cluster vectorizer.type=word2vec classifier.type=lr
```
or even change hyperparameters

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

While it's possible to override any configuration directly from the command line, the commands can quickly become long and hard to manage if a lot of parameters needs to be changed. So, for major updates such as changing the classifier or vectorizer parameters, or modifying grid search settings, it's much more convenient to open the existing YAML configuration file and update it manually.

**YAML config file**:  `config/baseline_config.yaml`

Once updated, you can run the script without needing any command-line overrides:

```bash
python scripts/run_baseline.py
```

For details on configuration, see [Baseline Configuration Explained](#-baseline-configuration-explained)


### 2. Prompting

**Entry Point**: `scripts/run_prompting.py`  
**Main Source Code**: `src/prompting/ollama_prompting.py`
**Prompts JSON**: `src/prompting/prompts/`

This script runs zero-shot, few-shot, or custom prompting experiments.

Before running an experiment, make sure to open a terminal and start the Ollama service by running:

```bash
ollama serve
```
**Examples:**

```bash
python scripts/run_prompting.py model=llama task=cluster prompt=few_shot
```

```bash
python scripts/run_prompting.py model=llama task=cluster prompt=custom
```

```bash
python scripts/run_prompting.py model=llama task=safety prompt=few_shot
```
**Note:** prompt=custom can be used only when the task=cluster

For details on prompt formatting and task-specific setup, see [Prompting Configuration Explained](#prompting-configuration-explained)

**Output Format**

Results are saved to  
`results/{task}/{prompt}/{model}/`

Files you will find there:

- `metrics.json` – accuracy, precision, recall, F1 (overall & per‑class)  
- `conf_matrix.png` – confusion‑matrix heatmap  
- `predictions.json` – list of inputs with the model’s raw output and final label

**Example:**

`results/cluster/few_shot/llama/llama_few_shot_confusion_matrix.png`
`results/cluster/few_shot/llama/llama_few_shot_metrics.json`
`results/cluster/few_shot/llama/llama_few_shot.json`

### 3. Fine-Tuning and Inference

⚠️ **IMPORTANT** Fine-tuning models for safety classification can take 10-24 hours, while dietary topic classification can range from 1-3 hours depending on parameter choice. Large batch sizes can throw memeory erros, because of not enough VRAM. 

### 🧪 Weights & Biases (W&B) Logging

When you run fine-tuning or evaluation scripts, you may see the following prompt in the terminal:

```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```


#### What to do:

- **Option 1**: Creates a new W&B account (you'll be redirected to a browser).
- **Option 2**: If you already have a W&B API token or want to use one (e.g., a shared token), choose this. It will prompt you to paste your API key.
- **Option 3**: Disables logging and just runs the script locally. Choose this if you don’t want W&B logs.

> ✅ If you're not interested in logging, **choose option 3** and everything will run as expected.


**Fine-Tuning Script**: `scripts/run_finetuning.py`  
Fine-tunes an instruction-tuned model like LLaMA 3 or Mistral using QLoRA and Hugging Face PEFT.

The fine-tuning setup is defined in `config/finetune_config.yaml`. The **three most important parameters** you may want to change are:

```yaml
task: "cluster"              # Choose "safety" or "cluster"
model_id: "llama3"           # Choose from: "llama3", "phi", "mistral"
model_type: "classification" # Choose "classification" or "causal"
```

These three control:
- Which dataset will be used (`task`)
- Which LLM is being fine-tuned (`model_id`)
- Whether classification or causal modeling is used (`model_type`)

**Example:**

```bash
python scripts/run_finetuning.py model_id=llama3 task=cluster model_type=classification
```

```bash
python scripts/run_finetuning.py model_id=phi task=safety 
```

These parameters can be controlled as well. You can edit them directly in `config/finetune_config.yaml` or override them from command line.

```yaml
sft_config:
  batch_size: 16                  
  epochs: 10                      
  lr: 2e-5                        
  evaluation_strategy: "epoch"   
  gradient_accumulation_steps: 2 
  max_grad_norm: 0.3             
  lr_scheduler_type: "cosine"    
  warmup_ratio: 0.05             
  optim: "paged_adamw_32bit"     
  logging_steps: 10              
  max_seq_length: 512            
  weight_decay: 0.005            
  loss_function: "custom"        

lora_config:
  r: 8
  alpha: 16
  target_modules: "all-linear" 
  dropout: 0.05
```

### Loss Function 

The `loss_function` field in the `sft_config` section determines how the model evaluates prediction errors during training. It supports two options:

#### `"standard"`
- **Description**: Uses the default *Cross-Entropy Loss* provided by Hugging Face Transformers.
- **Behavior**: Treats all class labels equally during optimization.
- **Use Case**: Best when your dataset is relatively balanced or you don't need to adjust for class imbalance.

####  `"custom"`
- **Description**: Uses a **weighted cross-entropy loss**, where class weights are automatically computed as the inverse frequency of each label in the training dataset.

> ⚠️ Note: The `loss_function` setting is **only applied when `model_type=classification`**. For causal language modeling tasks, loss behavior follows standard language modeling loss.


**Example:**

If parameters are modified from the config file then it's enough to run`

```bash
python scripts/run_finetuning.py model_id=mistral task=cluster sft_config.loss_function=standard sft_config.lr=1e-5
```

Otherwise, params can be overriden the following ways`

### Output Format and Run Name

All fine-tuned models, metrics, and logs are saved under the following directory:

```
results/finetuning/{model_id}/{task}/{run_name}/
```

####  Run Name Format
Each run is uniquely identified by a name encoding key training parameters:

```
llama3-cluster-classification-lr2e-05-bs16-20250715_182741
```


This tells you:
- `llama3`: Model ID used
- `cluster`: Classification task
- `classification`: Model type (as opposed to causal)
- `lr2e-05`: Learning rate was 2e-5
- `bs16`: Batch size was 16
- `20250715_182741`: Timestamp when the run started

This naming convention ensures that results are organized and easily traceable to the experiment's configuration.

###  Inference

**Script**: `scripts/run_inference.py`  
Runs prediction and evaluation using the saved adapter weights from a previous fine-tuning run.

#### 🔧 Important Parameter: `run_name`

The `run_name` identifies the exact fine-tuning run and should match the folder created under:

```
results/finetuning/{model_id}/{task}/{run_name}
```


It is automatically generated during fine-tuning using the pattern:

```
{model_id}-{task}-{model_type}-lr{learning_rate}-bs{batch_size}-{timestamp}
```

You must pass the correct `run_name` from a previous training job to ensure the model loads the right adapter and saves predictions in the correct folder.

#### ✅ Example:

```bash
python scripts/run_inference.py task=cluster model_id=llama3 run_name=llama3-cluster-classification-lr2e-05-bs16-20250706_214753
```

---

## Baseline Configuration Explained

The configuration file defines all parameters needed to run baseline classification experiments using traditional machine learning models and various feature extraction techniques.

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

#### Vectorizer Settings

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

#### Classifier Settings

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


#### Embedding and Model Paths

```yaml
word2vec_path: data/GoogleNews-vectors-negative300.bin
sentence_bert_model: sentence-transformers/all-MiniLM-L6-v2
```

- **`word2vec_path`**: Path to the Google News `.bin` Word2Vec file. Required only if `vectorizer.type` is `"word2vec"`.  
  See [Download Pre-trained Word2Vec Embeddings](#a-baseline-classification-setup) for instructions.

- **`sentence_bert_model`**: Name of the Sentence-BERT model hosted on Hugging Face (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).  
  Used only when `vectorizer.type` is set to `"sentence_bert"`. The model will be automatically downloaded and cached by Hugging Face's `transformers` library during the first run.

---

### Prompting Configuration Explained

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

####  `model`

Specifies which model to use for prompting. Options:
- `llama`: Meta LLaMA 3.1 8B
- `mistral`: Mistral 7B
- `phi`: Phi-3 Mini

The actual model string is resolved using the `ollama_models` mapping.

#### `task`

Defines the classification task to run. Options:
- `safety`: Classifies AI responses as "Safe" or "Unsafe"
- `cluster`: Classifies user struggles into one of 12 dietary topics

This value is used to load the appropriate dataset, prompt file, and output path.

#### `prompt`

Selects the prompting strategy. Options:
- `zero_shot`
- `few_shot`
- `custom`

Each prompt type changes how the model is guided during inference (e.g., examples included or not). Important to note that **custom** prompting works only with **cluster classification**.

####  Dataset Configuration

The dataset path is selected based on the `task`:

- `safety`: uses `data/processed/safety-dataset-90-5-5.hf`
- `cluster`: uses `data/processed/cluster-dataset-70-25-5.hf`

This is resolved dynamically via the `dataset_map`.

#### Ollama Model Mapping

Maps each model key (e.g., `llama`, `mistral`) to its actual name used by Ollama:

- `llama`: `llama3.1:8b`
- `mistral`: `mistral:7b`
- `phi`: `phi3:mini`

The final model loaded is stored in `ollama_model`.


#### Prompt Templates

The path to the prompt definitions (in JSON) is task-specific:

- For `safety`: `src/prompting/prompts/safety_prompts.json`
- For `cluster`: `src/prompting/prompts/cluster_prompts.json`

These files contain the actual input templates used in prompting (zero-shot, few-shot, or custom).

#### Output Directory

Results will be saved under a structured folder based on task, prompt type, and model:

- For `safety`: `results/safety/{prompt}/{model}`
- For `cluster`: `results/cluster/{prompt}/{model}`

Each directory contains model predictions, formatted outputs, and optionally confidence scores if available.

#### Label Names

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

