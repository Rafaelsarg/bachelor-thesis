import sys
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk
from sklearn.metrics import classification_report

# === Adjust import path for src/ ===
sys.path.append("src")

from finetuning.inference import GenericInference
# Formatter imports
from finetuning.formatters.llama_formatter import (
    LlamaSafetyPromptFormatterCasual,
    LlamaTopicPromptFormatterCausal,
    LlamaSafetyPromptFormatterClassification,
    LlamaTopicPromptFormatterClassification,
)
from finetuning.formatters.mistral_formatter import (
    MistralSafetyPromptFormatterCasual,
    MistralTopicPromptFormatterCausal,
    MistralSafetyPromptFormatterClassification,
    MistralTopicPromptFormatterClassification,
)
from finetuning.formatters.phi_formatter import (
    PhiSafetyPromptFormatterCasual,
    PhiTopicPromptFormatterCasual,
    PhiSafetyPromptFormatterClassification,
    PhiTopicPromptFormatterClassification,
)

# ─────────────────────────────────────────────
# 📦 Formatter Registry
# ─────────────────────────────────────────────
FORMATTER_REGISTRY = {
    "llama3": {
        "safety": {
            "casual": LlamaSafetyPromptFormatterCasual,
            "classification": LlamaSafetyPromptFormatterClassification,
        },
        "cluster": {
            "casual": LlamaTopicPromptFormatterCausal,
            "classification": LlamaTopicPromptFormatterClassification,
        },
    },
    "mistral": {
        "safety": {
            "casual": MistralSafetyPromptFormatterCasual,
            "classification": MistralSafetyPromptFormatterClassification,
        },
        "cluster": {
            "casual": MistralTopicPromptFormatterCausal,
            "classification": MistralTopicPromptFormatterClassification,
        },
    },
    "phi": {
        "safety": {
            "casual": PhiSafetyPromptFormatterCasual,
            "classification": PhiSafetyPromptFormatterClassification,
        },
        "cluster": {
            "casual": PhiTopicPromptFormatterCasual,
            "classification": PhiTopicPromptFormatterClassification,
        },
    },
}

def get_prompt_formatter(model_id: str, task: str, model_type: str):
    try:
        return FORMATTER_REGISTRY[model_id][task][model_type]()
    except KeyError:
        raise ValueError(f"Unsupported combination: model_id={model_id}, task={task}, model_type={model_type}")


# ─────────────────────────────────────────────
# Inference Runner
# ─────────────────────────────────────────────
@hydra.main(config_path="../config", config_name="inference_config", version_base=None)
def main(cfg: DictConfig):
    base_model_name = cfg.model_map[cfg.model_id]
    dataset = load_from_disk(cfg.dataset_path)["test"]

    # Select formatter based on model_id, task, and model_type
    formatter = get_prompt_formatter(cfg.model_id, cfg.task, cfg.model_type)

    # Load label mappings
    label2id = OmegaConf.to_container(cfg.label2id_map[cfg.task], resolve=True)
    id2label = {v: k for k, v in label2id.items()}

    # Initialize inference engine
    inference = GenericInference(
        base_model_name=base_model_name,
        adapter_dir=cfg.adapter_dir,
        prompt_formatter=formatter,
        model_type=cfg.model_type,
        hf_token=cfg.hf_token,
        label2id=label2id,
        id2label=id2label
    )

    # Run predictions
    predictions, references = inference.predict_dataset(dataset)

    # Remap safety labels for readability
    if cfg.task == "safety":
        label_map = {"Y": "Safe", "N": "Unsafe"}
        references = [label_map.get(r, r) for r in references]

    # Generate classification report
    report = classification_report(references, predictions, output_dict=True)

    # Save predictions
    with open(cfg.output_file, "w") as f:
        json.dump([
            {"text": ex["struggle"], "label": ref, "prediction": pred}
            for ex, ref, pred in zip(dataset, references, predictions)
        ], f, indent=2)

    # Save metrics
    with open(cfg.results_file, "w") as f:
        json.dump(report, f, indent=2)

    print("[✓] Inference complete. Metrics and predictions saved.")


if __name__ == "__main__":
    main()