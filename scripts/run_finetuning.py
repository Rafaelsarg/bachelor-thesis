import sys
import hydra
from omegaconf import DictConfig

# === Adjust import path for src/ ===
sys.path.append("src")

# === Generic trainer ===
from finetuning.trainer import GenericTrainer

# === Import all formatter classes ===
from finetuning.formatters.mistral_formatter import MistralSafetyPromptFormatter,  MistralTopicPromptFormatter
from finetuning.formatters.llama_formatter import LlamaSafetyPromptFormatter, LlamaTopicPromptFormatter
from finetuning.formatters.phi_formatter import PhiSafetyPromptFormatter, PhiTopicPromptFormatter

# === Nested formatter registry: model_id -> task -> formatter ===
FORMATTER_REGISTRY = {
    "mistral": {
        "safety": MistralSafetyPromptFormatter,
        "cluster": MistralTopicPromptFormatter
    },
    "llama3": {
        "safety": LlamaSafetyPromptFormatter,
        "cluster": LlamaTopicPromptFormatter
    },
    "phi": {
        "safety": PhiSafetyPromptFormatter,
        "cluster": PhiTopicPromptFormatter
    }
}

@hydra.main(config_path="../config", config_name="finetune_config", version_base=None)
def main(cfg: DictConfig):
    model_id = cfg.model_id
    task = cfg.task
    model_name = cfg.model_map[model_id]

    try:
        formatter_class = FORMATTER_REGISTRY[model_id][task]
    except KeyError:
        raise ValueError(
            f"No formatter found for model '{model_id}' and task '{task}'. "
            f"Available: {list(FORMATTER_REGISTRY.get(model_id, {}).keys())}"
        )

    formatter = formatter_class()

    trainer = GenericTrainer(
        model_name=model_name,
        dataset_path=cfg.dataset_path,
        output_dir=cfg.output_dir,
        cfg=cfg,
        prompt_formatter=formatter
    )

    trainer.train()

if __name__ == "__main__":
    main()
