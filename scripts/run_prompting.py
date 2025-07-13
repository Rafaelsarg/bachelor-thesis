from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf
import hydra
import ollama
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import sys

# === Adjust import path for src/ ===
sys.path.append("src")

from prompting.ollama_prompting import OllamaPrompting


def format_safety_prompt(text: str) -> str:
    """
    Preprocess the struggle+response string for safety classification.
    Assumes the relevant part is between #...# and #END#.
    """
    import re
    match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", text)
    if not match:
        return text
    struggle = match.group(1).strip()
    response = match.group(3).strip()
    return f"Patient: {struggle}\nDoctor: {response}\nClassify the safety of the doctor's response."


def run_custom_prompt(prompter, user_input: str, prompt_config: dict) -> str | None:
    """
    Execute 4-step narrowing classification using predefined groups and definitions.
    Each step presents 3 definitions; the final step compares the 3 chosen + 1 from remaining.
    If no valid label is chosen at any point, returns None.
    """
    system_prompt = prompt_config["system_prompt"]
    definitions = prompt_config["definitions"]
    groups = prompt_config["groups"]

    selected_labels = []

    for i, group in enumerate(groups[:3]):
        options = {label: definitions[label] for label in group}
        user_prompt = build_group_prompt(user_input, options)
        messages = prompter.build_custom_prompt(
            custom_system_instruction=system_prompt,
            user_input_text=user_prompt,
            context_example={"text": ""}  
        )
        prediction = prompter.send_prompt_to_model(message_sequence=messages).strip().upper()

        if prediction not in group:
            print(f"[!] Step {i+1}: Invalid prediction '{prediction}' not in group {group}")
            return None
        selected_labels.append(prediction)

    remaining_group = [label for label in groups[3] if label not in selected_labels]
    final_group = selected_labels + ([remaining_group[0]] if remaining_group else [])

    options = {label: definitions[label] for label in final_group}
    user_prompt = build_group_prompt(user_input, options)
    messages = prompter.build_custom_prompt(
        custom_system_instruction=system_prompt,
        user_input_text=user_prompt,
        context_example={"text": ""}
    )
    prediction = prompter.send_prompt_to_model(message_sequence=messages).strip().upper()

    if prediction not in final_group:
        print(f"[!] Final step: Invalid prediction '{prediction}' not in final group {final_group}")
        return None

    return prediction


def build_group_prompt(user_input: str, label_definitions: dict) -> str:
    """
    Builds a prompt using square-bracketed input and a list of category definitions.
    """
    label_list = "\n".join([f"{label}: {desc}" for label, desc in label_definitions.items()])
    return f"[{user_input}]\n\n{label_list}"

def evaluate_and_log_metrics(results: list, labels: list[str], focus_label: str = None) -> dict:
    """
    Compute precision, recall, f1-score, and optionally focus on a specific label.
    
    Args:
        results (list): List of dicts with "actual" and "predicted" keys.
        labels (list): List of label names (e.g. ["Safe", "Unsafe"]).
        focus_label (str): If specified, filters metrics to that class.
    
    Returns:
        dict: Full classification report.
    """
    y_true = [r["actual"] for r in results]
    y_pred = [r["predicted"] for r in results]

    # Generate full report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    return report

def save_confusion_matrix(results: list, labels: list[str], output_path: str, title: str = ""):
    """
    Save a confusion matrix plot to a file.
    
    Args:
        results (list): List of dicts with "actual" and "predicted".
        labels (list): List of class labels in correct order.
        output_path (str): Where to save the PNG image.
        title (str): Optional title for the plot.
    """
    y_true = [r["actual"] for r in results]
    y_pred = [r["predicted"] for r in results]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title or "Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Confusion matrix saved to {output_path}")

@hydra.main(config_path="../config", config_name="prompting_config", version_base=None)
def main(cfg: DictConfig):
    # ------------ Configuration ------------
    dataset_cfg = OmegaConf.to_container(cfg.dataset_config, resolve=True)
    prompts_dir = cfg.prompts_path[cfg.task]
    output_dir = cfg.output_directory[cfg.task]
    os.makedirs(output_dir, exist_ok=True)

    # ------------ Load Data ------------
    data = DatasetDict.load_from_disk(dataset_cfg['dataset_path'])['test']    

    # ---------- Pull the Model ---------
    model = cfg.ollama_model
    ollama.pull(model)

    # ------------ Load Prompt Template ------------
    with open(prompts_dir) as f:
        prompt_template = json.load(f)[cfg.prompt]

    # ---- Instantiate Prompting Class ------
    print(prompt_template.keys())
    prompter = OllamaPrompting(model_name=model, system_instruction=prompt_template['system_prompt'])

    #------------ Run Prompting ------------
    results = []
    for sample in data:
        input_text = sample["struggle"]
        actual = sample["label"]

        if cfg.task == 'safety':
            input_text = format_safety_prompt(input_text)

        if cfg.prompt == "zero_shot":
            messages = prompter.build_zero_shot_prompt(user_input_text=input_text)
            predicted = prompter.send_prompt_to_model(message_sequence=messages)
        elif cfg.prompt == "few_shot":
            examples = prompt_template.get("few_shot_examples", [])
            messages = prompter.build_few_shot_prompt(user_input_text=input_text, example_pairs=examples)
            predicted = prompter.send_prompt_to_model(message_sequence=messages)
        elif cfg.prompt == "custom":
            predicted = run_custom_prompt(prompter, user_input=input_text, prompt_config=prompt_template)
        else:
            raise ValueError(f"Unsupported prompt type: {cfg.prompt}")

        results.append({
            "input": input_text,
            "actual": actual,
            "predicted": predicted
        })

        print(f"Input: {input_text}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}\n")

    # ------------ Save Results ------------
    output_file = os.path.join(output_dir, f"{cfg.model}_{cfg.prompt}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Evaluate and log metrics
    label_names = cfg.label_names[cfg.task]
    metrics = evaluate_and_log_metrics(results, labels=label_names, focus_label="Unsafe" if cfg.task == "safety" else None)
    with open(os.path.join(output_dir, f"{cfg.model}_{cfg.prompt}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    confmat_path = os.path.join(output_dir, f"{cfg.model}_{cfg.prompt}_confusion_matrix.png")
    save_confusion_matrix(results, labels=label_names, output_path=confmat_path)
    print(f"[✓] Results saved to: {output_file}")

if __name__ == "__main__":
    main()