from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    def __init__(self, base_model_name: str, adapter_dir: str, prompt_formatter):
        self.prompt_formatter = prompt_formatter
        self.hf_token = os.environ.get("HF_TOKEN", None)

        # Hugging Face auth (optional)
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
        self.tokenizer.model_max_length = 512
        print("[✓] Tokenizer loaded")

        # Load quantized model base (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config
        )

        # Load adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        print("[✓] LoRA adapter applied and model ready")

    @abstractmethod
    def predict(self, struggle: str):
        """
        Abstract method for making predictions on a single input.
        """
        pass
