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
from finetuning.formatters.mistral_formatter import MistralSafetyPromptFormatterCasual,  MistralSafetyPromptFormatterClassification, MistralTopicPromptFormatterCausal, MistralTopicPromptFormatterClassification
from finetuning.formatters.llama_formatter import LlamaSafetyPromptFormatterCasual, LlamaSafetyPromptFormatterClassification, LlamaTopicPromptFormatterCausal, LlamaTopicPromptFormatterClassification
from finetuning.formatters.phi_formatter import PhiSafetyPromptFormatterCasual, PhiSafetyPromptFormatterClassification, PhiTopicPromptFormatterCasual, PhiTopicPromptFormatterClassification

# === Nested formatter registry: model_id -> task -> formatter ===
FORMATTER_REGISTRY = {
    "mistral": {
        "safety": {
            "casual": MistralSafetyPromptFormatterCasual,
            "classification": MistralSafetyPromptFormatterClassification
        },
        "cluster": {
            "casual": MistralTopicPromptFormatterCausal,
            "classification": MistralTopicPromptFormatterClassification
        }
    },
    "llama3": {
        "safety": {
            "casual": LlamaSafetyPromptFormatterCasual,
            "classification": LlamaSafetyPromptFormatterClassification
        },
        "cluster": {
            "casual": LlamaTopicPromptFormatterCausal,
            "classification": LlamaTopicPromptFormatterClassification
        }
    },
    "phi": {
        "safety": {
            "casual": PhiSafetyPromptFormatterCasual,
            "classification": PhiSafetyPromptFormatterClassification
        },
        "cluster": {
            "casual": PhiTopicPromptFormatterCasual,
            "classification": PhiTopicPromptFormatterClassification
        }
    }
}

def save_full_config(cfg, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    config_dir = os.path.join(output_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # Save the full config as one file
    with open(os.path.join(config_dir, "full_config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    # Optionally split and save specific parts
    if "sft_config" in cfg and "sft_cfg" in cfg.sft_config:
        with open(os.path.join(config_dir, "sft_config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg.sft_config.sft_cfg, resolve=True), f, indent=2)

    if "lora_config" in cfg:
        with open(os.path.join(config_dir, "lora_config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg.lora_config, resolve=True), f, indent=2)

    if "dataset_config" in cfg:
        with open(os.path.join(config_dir, "dataset_config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg.dataset_config, resolve=True), f, indent=2)
                      
@hydra.main(config_path="../config", config_name="finetune_config", version_base=None)
def main(cfg: DictConfig):
    model_id = cfg.model_id
    task = cfg.task
    model_name = cfg.model_map[model_id]
    model_type = cfg.model_type

    try:
        formatter_class = FORMATTER_REGISTRY[model_id][task][model_type]
    except KeyError:
        raise ValueError(
            f"No formatter found for model '{model_id}' and task '{task}' and model type {model_type}. "
            f"Available: {list(FORMATTER_REGISTRY.get(model_id, {}).keys())}"
        )

    formatter = formatter_class()

    trainer = GenericTrainer(
        model_name=model_name,
        output_dir=cfg.output_dir,
        cfg=cfg,
        prompt_formatter=formatter
    )

    trainer.train()
    save_full_config(cfg, cfg.output_dir)

if __name__ == "__main__":
    main()