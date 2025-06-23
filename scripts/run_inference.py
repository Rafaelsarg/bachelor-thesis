import sys
import json
import hydra
from omegaconf import DictConfig
from datasets import load_from_disk
from sklearn.metrics import classification_report

# === Adjust import path for src/ ===
sys.path.append("src")

from finetuning.inference import GenericInference
from finetuning.formatters.llama_formatter import LlamaSafetyPromptFormatter, LlamaTopicPromptFormatter
from finetuning.formatters.mistral_formatter import MistralSafetyPromptFormatter, MistralTopicPromptFormatter
from finetuning.formatters.phi_formatter import PhiSafetyPromptFormatter, PhiTopicPromptFormatter


# ─────────────────────────────────────────────
# 🧠 Formatter Selector
# ─────────────────────────────────────────────
def get_prompt_formatter(model_id: str, task: str):
    if model_id == "llama3":
        return LlamaSafetyPromptFormatter() if task == "safety" else LlamaTopicPromptFormatter()
    elif model_id == "mistral":
        return MistralSafetyPromptFormatter() if task == "safety" else MistralTopicPromptFormatter()
    elif model_id == "phi":
        return PhiSafetyPromptFormatter() if task == "safety" else PhiTopicPromptFormatter()
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")


# ─────────────────────────────────────────────
# 🚀 Inference Runner
# ─────────────────────────────────────────────
@hydra.main(config_path="../config", config_name="inference_config", version_base=None)
def main(cfg: DictConfig):
    base_model_name = cfg.model_map[cfg.model_id]
    dataset = load_from_disk(cfg.dataset_path)["test"]

    # Prompt Formatter and Inference
    formatter = get_prompt_formatter(cfg.model_id, cfg.task)
    inference = GenericInference(base_model_name, cfg.adapter_dir, formatter)

    predictions, references = inference.predict_dataset(dataset)

    if cfg.task == "safety":
        label_map = {"Y": "Safe", "N": "Unsafe"}
        references = [label_map.get(r, r) for r in references]

    # Classification report
    report = classification_report(references, predictions, output_dict=True)

    # Save predictions
    with open(cfg.output_file, "w") as f:
        json.dump([
            {"text": ex["struggle"], "label": ref, "prediction": pred}
            for ex, ref, pred in zip(dataset, references, predictions)
        ], f, indent=2)

    # Save results
    with open(cfg.results_file, "w") as f:
        json.dump(report, f, indent=2)

    print("[✓] Inference complete. Metrics saved.")


if __name__ == "__main__":
    main()
