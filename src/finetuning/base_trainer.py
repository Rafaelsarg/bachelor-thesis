from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import DatasetDict, ClassLabel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from transformers.utils.quantization_config import BitsAndBytesConfig
from omegaconf import OmegaConf
from collections import Counter
import wandb
import torch
import os
import math


class BasePromptFormatter(ABC):
    @abstractmethod
    def format_prompt_training(self, struggle: str, label: str) -> dict:
        pass

    @abstractmethod
    def format_prompt_inference(self, struggle: str) -> str:
        pass

class BaseTrainer(ABC):
    def __init__(self, model_name: str,  output_dir: str, cfg, prompt_formatter: BasePromptFormatter):
        # 1. Config
        self.model_name = model_name
        self.output_dir = output_dir
        self.prompt_formatter = prompt_formatter

        self.sft_cfg = OmegaConf.to_container(cfg.sft_config, resolve=True)
        self.lora_cfg = OmegaConf.to_container(cfg.lora_config, resolve=True)
        self.dataset_cfg = OmegaConf.to_container(cfg.dataset_config, resolve=True)
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

        # 4. Model type
        self.model_type = getattr(cfg, "model_type", "classification")

        # 5. Loss function
        self.loss_function = getattr(cfg, "loss_function", "custom")

        wandb.init(
            project=getattr(cfg, "project_name", "llm-finetuning"),
            name=getattr(cfg, "run_name", None),
            config={
                "model_name": model_name,
                "model_type": self.model_type,
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
        self.tokenizer.model_max_length = self.sft_cfg.get("max_seq_length", 512) 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("[✓] Tokenizer loaded")

        # 6. Load model w/ QLoRA
        self.model = self._load_and_prepare_model()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id       
        print("[✓] Model loaded & LoRA applied")

        # 7. Load & tokenize dataset
        self.dataset_path = self.dataset_cfg["dataset_path"]
        self.dataset = self._load_and_format_dataset()
        print(self.dataset)
        print("[✓] Dataset processed")
        # 8. Compute class weights if needed
        if self.model_type == "classification":
            self.class_weights = self._compute_class_weights(
                dataset_split="train",
                label_column="label"
            )
            print(f"[✓] Class weights computed: {self.class_weights.tolist()}")

    def _load_and_prepare_model(self):
        # Configure 4-bit quantization using BitsAndBytes for QLoRA.
        # This drastically reduces memory usage and enables large model training on limited hardware.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load the base causal or classification language model with quantization applied.
        # `device_map="auto"` allows Hugging Face to assign the model to available GPU(s).
        # `trust_remote_code=True` is necessary for some custom model architectures.
        if self.model_type == "casual":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            task_type = "CAUSAL_LM"
        elif self.model_type == "classification":
            label2id = self.dataset_cfg["label2id"]
            id2label = {v: k for k, v in label2id.items()}

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                num_labels=self.dataset_cfg['num_labels'],  # Make this dynamic later if needed
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                id2label=id2label,
                label2id=label2id
            )
            task_type = "SEQ_CLS"
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Create the LoRA configuration specifying which modules to fine-tune
        # and how the low-rank adaptation should be applied.
        lora_config = LoraConfig(
            r=self.lora_cfg["r"],
            lora_alpha=self.lora_cfg["alpha"],
            target_modules=self.lora_cfg["target_modules"],
            lora_dropout=self.lora_cfg["dropout"],
            bias="none",
            task_type=task_type
        )
        model.enable_input_require_grads()
        model = prepare_model_for_kbit_training(model)

        return get_peft_model(model, lora_config)

    def _load_and_format_dataset(self):
        dataset = DatasetDict.load_from_disk(self.dataset_path)

        def process(example):
            result = self.prompt_formatter.format_prompt_training(example["struggle"], example["label"])
            return result if result is not None else {}

        # Apply prompt formatting
        dataset = dataset.map(
            process,
            remove_columns=["struggle"],
            load_from_cache_file=False
        )

        # For classification, cast label column and keep labels
        if self.model_type == "classification":
            dataset = dataset.cast_column(
                "label",
                ClassLabel(
                    num_classes=self.dataset_cfg["num_labels"],
                    names=self.dataset_cfg["label_names"]
                )
            )   


        # Determine which columns to remove
        remove_cols = ["text"]
        if self.model_type == "causal":
            # causal LM doesn't need "label"
            remove_cols.append("label")

        # Tokenize
        dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=remove_cols,
        )

        return dataset

    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )

    def _compute_class_weights(self, dataset_split="train", label_column="label") -> torch.Tensor:
        """
        Computes class weights based on inverse frequency in the specified dataset split.

        Args:
            dataset_split (str): Which split to use ("train", "validation", etc.)
            label_column (str): Column name where class labels are stored.

        Returns:
            torch.Tensor: Tensor of class weights (dtype=torch.float32).
        """
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Make sure to load it before computing class weights.")

        labels = self.dataset[dataset_split][label_column]

        label_counts = Counter(labels)
        total = sum(label_counts.values())
        num_classes = len(label_counts)

        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)  # use 1 to avoid division by zero
            weight = total / (num_classes * count)  # standard inverse frequency formula
            weights.append(weight)

        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        return weight_tensor
    
    @abstractmethod
    def train(self):
        pass
