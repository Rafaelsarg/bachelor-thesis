# Thesis Codebase: Dietary Topic and Safety Classification

This repository contains the codebase developed for thesis experiments involving dietary topic classification and safety assessment of AI-generated nutrition counseling.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running Baseline Models](#running-baseline-models)
- [Fine-Tuning Large Language Models](#fine-tuning-large-language-models)
- [Prompting with Ollama](#prompting-with-ollama)
- [Configuration and Parameters](#configuration-and-parameters)
- [Results and Logging](#results-and-logging)
- [License](#license)

---

### Prerequisites

To run the codebase successfully, ensure that the following software and accounts are available and properly configured:

- **Python 3.12.3**  
  The project requires Python version 3.12.3. It is recommended to use a virtual environment (e.g., `venv` or `conda`) to avoid conflicts with other packages.

- **NVIDIA GPU: A40 or A100 (tested on Linux machines)**  
  Training and fine-tuning large language models is computationally intensive. The code was tested on Linux machines equipped with NVIDIA A40 and A100 GPUs. Other CUDA-compatible GPUs may also work, but performance and compatibility are not guaranteed.

- **CUDA-compatible GPU drivers**  
  Ensure that the correct NVIDIA drivers and CUDA toolkit are installed on your system. These are necessary for PyTorch and Hugging Face Transformers to leverage GPU acceleration.

- **[Hugging Face](https://huggingface.co) account and access token**  
  Required to download and use pre-trained models from the Hugging Face Hub (e.g., Mistral, Phi, LLaMA).  
  A token will be provided for reproduction purposes. Alternatively, you can [sign up for a free Hugging Face account](https://huggingface.co/join) and [generate your own access token](https://huggingface.co/settings/tokens) with at least `read` permission.

  ⚠️ **For LLaMA models** (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`), you must first request access from Meta. Visit the [LLaMA model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and click **"Access repository"**. You must be logged in to your Hugging Face account. Approval from Meta is required before you can download or use the model.

- **[Weights & Biases](https://wandb.ai/) account and API key**  
  Used for experiment tracking and logging. The code automatically logs metrics (e.g., loss, accuracy, F1) and configurations to Weights & Biases. A pre-configured token will be provided, or you can [register for your own](https://wandb.ai/authorize).

- **[Ollama](https://ollama.com/)**  
  Required for prompting tasks that use locally hosted LLMs (e.g., Mistral, LLaMA, Phi). Install Ollama following the instructions on their website, and make sure it is running before launching any prompting scripts.


---

### Project Structure

The codebase is organized for modularity, reproducibility, and clarity. Below is an overview of the main folders and their roles:


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

```bash
git clone <repository-url>
cd bachelor-thesis
```

#### 2. Create a Virtual Environment

**Using venv (recommended):**
```bash
python3.12 -m venv venv
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

#### 4. Verify CUDA Installation

Ensure CUDA is properly installed and accessible:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

If CUDA is not available, you may need to:
- Install NVIDIA drivers
- Install CUDA toolkit
- Reinstall PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

#### 5. Task-Specific Setup

The codebase supports three main types of experiments. Follow the setup instructions for the tasks you plan to run:

##### A. Baseline Classification Setup

**Required for:** Traditional ML classification experiments

**Setup:**
```bash
# Verify scikit-learn installation
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

# Test baseline imports
python -c "from src.baselines.classification import BaselineClassifier; print('Baseline module imported successfully!')"
```

**Configuration:**
- Review `config/baseline_config.yaml` for model parameters
- Ensure datasets are available in `data/processed/`

##### B. Fine-tuning Setup

**Required for:** Fine-tuning large language models

**Setup:**
```bash
# Configure Hugging Face token
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Configure Weights & Biases
export WANDB_API_KEY="your_wandb_api_key_here"

# Test fine-tuning imports
python -c "from src.finetuning.trainer import FineTuningTrainer; print('Fine-tuning module imported successfully!')"
```

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

**Pull required models:**
```bash
# For prompting experiments
ollama pull mistral:7b
ollama pull llama3.2:3b
ollama pull phi:latest
```

**Test Ollama connection:**
```bash
python -c "import requests; print('Ollama status:', requests.get('http://localhost:11434/api/tags').status_code)"
```

**Configuration:**
- Review `config/prompting_config.yaml` for model and prompt settings
- Ensure prompts are available in `src/prompting/prompts/`

#### 6. Verify Complete Installation

Run these commands to verify all components:

```bash
# Test Python imports
python -c "import torch, transformers, datasets, wandb, hydra; print('✓ Core packages imported')"

# Test CUDA
python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Test baseline setup
python -c "from src.baselines.classification import BaselineClassifier; print('✓ Baseline setup complete')"

# Test fine-tuning setup
python -c "from src.finetuning.trainer import FineTuningTrainer; print('✓ Fine-tuning setup complete')"

# Test prompting setup
python -c "import requests; status = requests.get('http://localhost:11434/api/tags').status_code; print(f'✓ Ollama status: {status}')"
```

#### 7. Download Datasets

The processed datasets should be available in `data/processed/`:
- `cluster-dataset-70-25-5.hf` - For topic classification
- `safety-dataset-90-5-5.hf` - For safety assessment

If datasets are missing, contact the repository maintainer for access.

#### Troubleshooting

**Common Issues:**

1. **CUDA out of memory errors:**
   - Reduce batch sizes in configuration files
   - Use gradient accumulation
   - Consider using smaller models

2. **Hugging Face token errors:**
   - Ensure your token has the correct permissions
   - For LLaMA models, verify you have access granted by Meta

3. **Ollama connection issues:**
   - Ensure Ollama service is running: `ollama serve`
   - Check if port 11434 is available
   - Verify model downloads: `ollama list`

4. **Package conflicts:**
   - Use a fresh virtual environment
   - Install packages in the order specified in requirements.txt
   - Consider using `pip install --no-deps` for problematic packages

**Getting Help:**
- Check the [Issues](link-to-issues) page for known problems
- Review the configuration files in `config/` for parameter adjustments
- Contact the maintainer for dataset access or model permissions




