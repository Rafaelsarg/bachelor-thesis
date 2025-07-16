from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from transformers.utils.quantization_config import BitsAndBytesConfig
from huggingface_hub import login
import torch
import os


class BaseInference(ABC):
    """
    Base class for handling inference and evaluation.
    Loads the tokenizer, base model, and LoRA adapter.
    """

    def __init__(
            self, 
            base_model_name: str, 
            adapter_dir: str, 
            prompt_formatter,
            model_type: str,
            sequence_length: int = 512,
            hf_token: str = None,
            label2id: dict = None,
            id2label: dict = None
        ):
        self.prompt_formatter = prompt_formatter
        self.model_type = model_type
        self.label2id = label2id
        self.id2label = id2label
        self.hf_token = hf_token

        if self.hf_token:
            login(token=self.hf_token)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[✓] Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=self.hf_token,
            use_fast=False,
            local_files_only=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = sequence_length
        print("[✓] Tokenizer loaded")

        # Load quantized model base (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        if self.model_type == "causal":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                quantization_config=bnb_config
            )
        elif self.model_type == "classification":
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                num_labels=len(self.label2id),
                label2id=self.label2id,
                id2label=self.id2label
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Load adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        print("[✓] LoRA adapter applied and model ready")

    @abstractmethod
    def predict(self, struggle: str):
        """
        Abstract method for making predictions on a single input.
        """
        pass
