import sys
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Adjust import path for src/ ===
sys.path.append("src")

from finetuning.inference import GenericInference
# Formatter imports
from finetuning.formatters.llama_formatter import (
    LlamaSafetyPromptFormatterCausal,
    LlamaTopicPromptFormatterCausal,
    LlamaSafetyPromptFormatterClassification,
    LlamaTopicPromptFormatterClassification,
)
from finetuning.formatters.mistral_formatter import (
    MistralSafetyPromptFormatterCausal,
    MistralTopicPromptFormatterCausal,
    MistralSafetyPromptFormatterClassification,
    MistralTopicPromptFormatterClassification,
)
from finetuning.formatters.phi_formatter import (
    PhiSafetyPromptFormatterCausal,
    PhiTopicPromptFormatterCausal,
    PhiSafetyPromptFormatterClassification,
    PhiTopicPromptFormatterClassification,
)

# ─────────────────────────────────────────────
# 📦 Formatter Registry
# ─────────────────────────────────────────────
FORMATTER_REGISTRY = {
    "llama": {
        "safety": {
            "causal": LlamaSafetyPromptFormatterCausal,
            "classification": LlamaSafetyPromptFormatterClassification,
        },
        "cluster": {
            "causal": LlamaTopicPromptFormatterCausal,
            "classification": LlamaTopicPromptFormatterClassification,
        },
    },
    "mistral": {
        "safety": {
            "causal": MistralSafetyPromptFormatterCausal,
            "classification": MistralSafetyPromptFormatterClassification,
        },
        "cluster": {
            "causal": MistralTopicPromptFormatterCausal,
            "classification": MistralTopicPromptFormatterClassification,
        },
    },
    "phi": {
        "safety": {
            "causal": PhiSafetyPromptFormatterCausal,
            "classification": PhiSafetyPromptFormatterClassification,
        },
        "cluster": {
            "causal": PhiTopicPromptFormatterCausal,
            "classification": PhiTopicPromptFormatterClassification,
        },
    },
}

def get_prompt_formatter(model_id: str, task: str, model_type: str):
    try:
        return FORMATTER_REGISTRY[model_id][task][model_type]()
    except KeyError:
        raise ValueError(f"Unsupported combination: model_id={model_id}, task={task}, model_type={model_type}")

def save_confusion_matrix(references, predictions, labels, output_dir: str):
    """
    Save a confusion matrix plot using seaborn.
    """
    cm = confusion_matrix(references, predictions, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    pic_dir = os.path.join(output_dir, "pic")
    os.makedirs(pic_dir, exist_ok=True)
    cm_path = os.path.join(pic_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[✓] Confusion matrix saved at: {cm_path}")

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
        sequence_length=cfg.sequence_length,
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
    report = classification_report(references, predictions, output_dict=True, zero_division=0)

    # Save predictions
    with open(cfg.output_file, "w") as f:
        json.dump([
            {"text": ex["struggle"], "label": ref, "prediction": pred}
            for ex, ref, pred in zip(dataset, references, predictions)
        ], f, indent=2)

    # Save metrics
    with open(cfg.results_file, "w") as f:
        json.dump(report, f, indent=2)

    # Save confusion matrix
    save_confusion_matrix(
        references=references,
        predictions=predictions,
        labels=list(id2label.values()),
        output_dir=cfg.images_dir
    )

    print("[✓] Inference complete. Metrics and predictions saved.")


if __name__ == "__main__":
    main()