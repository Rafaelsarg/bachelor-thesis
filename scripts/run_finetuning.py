import sys
import hydra
from omegaconf import DictConfig

# === Adjust import path for src/ ===
sys.path.append("src")

# === Generic trainer ===
from finetuning.trainer import GenericTrainer

# === Import all formatter classes ===
from finetuning.formatters.mistral_formatter import MistralSafetyPromptFormatterCasual,  MistralSafetyPromptFormatterClassification, MistralTopicPromptFormatter
from finetuning.formatters.llama_formatter import LlamaSafetyPromptFormatterCasual, LlamaSafetyPromptFormatterClassification, LlamaTopicPromptFormatter
from finetuning.formatters.phi_formatter import PhiSafetyPromptFormatterCasual, PhiSafetyPromptFormatterClassification, PhiTopicPromptFormatter

# === Nested formatter registry: model_id -> task -> formatter ===
FORMATTER_REGISTRY = {
    "mistral": {
        "safety": {
            "causal": MistralSafetyPromptFormatterCasual,
            "classification": MistralSafetyPromptFormatterClassification
        },
        "cluster": {
            "causal": MistralTopicPromptFormatter,
        }
    },
    "llama3": {
        "safety": {
            "causal": LlamaSafetyPromptFormatterCasual,
            "classification": LlamaSafetyPromptFormatterClassification
        },
        "cluster": {
            "causal": LlamaTopicPromptFormatter,
        }
    },
    "phi": {
        "safety": {
            "causal": PhiSafetyPromptFormatterCasual,
            "classification": PhiSafetyPromptFormatterClassification
        },
        "cluster": {
            "causal": PhiTopicPromptFormatter,
        }
    }
}

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

if __name__ == "__main__":
    main()
