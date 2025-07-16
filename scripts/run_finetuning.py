import sys
import hydra
import os
import json
from omegaconf import DictConfig, OmegaConf

# === Adjust import path for src/ ===
sys.path.append("src")

# === Generic trainer ===
from finetuning.trainer import GenericTrainer

# === Import all formatter classes ===
from finetuning.formatters.mistral_formatter import MistralSafetyPromptFormatterCausal,  MistralSafetyPromptFormatterClassification, MistralTopicPromptFormatterCausal, MistralTopicPromptFormatterClassification
from finetuning.formatters.llama_formatter import LlamaSafetyPromptFormatterCausal, LlamaSafetyPromptFormatterClassification, LlamaTopicPromptFormatterCausal, LlamaTopicPromptFormatterClassification
from finetuning.formatters.phi_formatter import PhiSafetyPromptFormatterCausal, PhiSafetyPromptFormatterClassification, PhiTopicPromptFormatterCausal, PhiTopicPromptFormatterClassification

# === Nested formatter registry: model_id -> task -> formatter ===
FORMATTER_REGISTRY = {
    "mistral": {
        "safety": {
            "causal": MistralSafetyPromptFormatterCausal,
            "classification": MistralSafetyPromptFormatterClassification
        },
        "cluster": {
            "causal": MistralTopicPromptFormatterCausal,
            "classification": MistralTopicPromptFormatterClassification
        }
    },
    "llama3": {
        "safety": {
            "causal": LlamaSafetyPromptFormatterCausal,
            "classification": LlamaSafetyPromptFormatterClassification
        },
        "cluster": {
            "causal": LlamaTopicPromptFormatterCausal,
            "classification": LlamaTopicPromptFormatterClassification
        }
    },
    "phi": {
        "safety": {
            "causal": PhiSafetyPromptFormatterCausal,
            "classification": PhiSafetyPromptFormatterClassification
        },
        "cluster": {
            "causal": PhiTopicPromptFormatterCausal,
            "classification": PhiTopicPromptFormatterClassification
        }
    }
}


def save_full_config(cfg, output_dir: str):
    config_dir = os.path.join(output_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, "full_config.json")
    with open(config_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
                      
# ---------------- Hydra entry point for configuration loading ----------------
@hydra.main(config_path="../config", config_name="finetune_config", version_base=None)
def main(cfg: DictConfig):
    # ---------------- Safety check: Raise error if using Phi model with classification ----------------
    if cfg.model_id == 'phi' and cfg.model_type == 'classification':
        raise ValueError("Phi model does not support classification head. Use causal model instead.")

    # ---------------- Load model and task settings from config ----------------
    model_id = cfg.model_id
    task = cfg.task
    model_name = cfg.model_map[model_id]
    model_type = cfg.model_type

    # ---------------- Retrieve correct prompt formatter from registry ----------------
    try:
        formatter_class = FORMATTER_REGISTRY[model_id][task][model_type]
    except KeyError:
        raise ValueError(
            f"No formatter found for model '{model_id}' and task '{task}' and model type {model_type}. "
            f"Available: {list(FORMATTER_REGISTRY.get(model_id, {}).keys())}"
        )

    # ---------------- Save the full Hydra configuration to the output directory ----------------
    save_full_config(cfg, cfg.output_dir)

    # ---------------- Initialize the selected formatter ----------------
    formatter = formatter_class()

    # ---------------- Initialize and run the training procedure ----------------
    trainer = GenericTrainer(
        model_name=model_name,
        output_dir=cfg.output_dir,
        cfg=cfg,
        prompt_formatter=formatter
    )

    trainer.train()

if __name__ == "__main__":
    main()
