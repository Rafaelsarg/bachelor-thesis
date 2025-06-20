from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from transformers.utils.quantization_config import BitsAndBytesConfig
from omegaconf import OmegaConf
import wandb
import torch
import os


class BasePromptFormatter(ABC):
    @abstractmethod
    def format_prompt_training(self, struggle: str, label: str) -> dict:
        pass

    @abstractmethod
    def format_prompt_inference(self, struggle: str) -> str:
        pass

class BaseTrainer(ABC):
    def __init__(self, model_name: str, dataset_path: str, output_dir: str, cfg, prompt_formatter: BasePromptFormatter):
        # 1. Config
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.prompt_formatter = prompt_formatter

        self.sft_cfg = OmegaConf.to_container(cfg.sft_config, resolve=True)
        self.lora_cfg = OmegaConf.to_container(cfg.lora_config, resolve=True)
        self.hf_token = cfg.hf_token if hasattr(cfg, "hf_token") else os.environ.get("HF_TOKEN")

        # 2. Output dirs
        self.model_output_dir = os.path.join(output_dir, "model")
        self.logs_output_dir = os.path.join(output_dir, "logs")
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.model_output_dir, exist_ok=True)
        os.makedirs(self.logs_output_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # 3. Hugging Face login
        if self.hf_token:
            login(token=self.hf_token)

        wandb.init(
            project=getattr(cfg, "project_name", "llm-finetuning"),
            name=getattr(cfg, "run_name", None),
            config={
                "model_name": model_name,
                "batch_size": self.sft_cfg["batch_size"],
                "learning_rate": self.sft_cfg["lr"],
                "epochs": self.sft_cfg["epochs"],
                "gradient_accumulation_steps": self.sft_cfg["gradient_accumulation_steps"],
                "target_modules": self.lora_cfg["target_modules"],
                "task": getattr(cfg, "task", "unknown")
            },
            dir=self.logs_output_dir
        )
        print("[✓] WandB initialized")

        # 4. Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[✓] Using device: {self.device}")

        # 5. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            use_fast=False,
            local_files_only=False
        )
        self.tokenizer.model_max_length = 512
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[✓] Tokenizer loaded")

        # 6. Load model w/ QLoRA
        self.model = self._load_and_prepare_model()
        print("[✓] Model loaded & LoRA applied")

        # 7. Load & tokenize dataset
        self.dataset = self._load_and_format_dataset()
        print("[✓] Dataset processed")

    def _load_and_prepare_model(self):
        # Configure 4-bit quantization using BitsAndBytes for QLoRA.
        # This drastically reduces memory usage and enables large model training on limited hardware.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load the base causal language model with quantization applied.
        # `device_map="auto"` allows Hugging Face to assign the model to available GPU(s).
        # `trust_remote_code=True` is necessary for some custom model architectures.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Create the LoRA configuration specifying which modules to fine-tune
        # and how the low-rank adaptation should be applied.
        lora_config = LoraConfig(
            r=self.lora_cfg["r"],
            lora_alpha=self.lora_cfg["alpha"],
            target_modules=self.lora_cfg["target_modules"],
            lora_dropout=self.lora_cfg["dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )

        return get_peft_model(model, lora_config)

    def _load_and_format_dataset(self):
        dataset = DatasetDict.load_from_disk(self.dataset_path)

        def process(example):
            return self.prompt_formatter.format_prompt_training(example["struggle"], example["label"])

        dataset = dataset.map(process, remove_columns=["struggle", "label"], load_from_cache_file=False)
        return dataset

    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    @abstractmethod
    def train(self):
        pass
