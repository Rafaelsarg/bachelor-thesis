from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf
import hydra
import ollama
import json
import os
import sys

# === Adjust import path for src/ ===
sys.path.append("src")

from prompting.ollama_prompting import OllamaPrompting
from prompting.utils import format_safety_prompt, evaluate_and_log_metrics, save_confusion_matrix

def run_custom_prompt(prompter, user_input: str, prompt_config: dict) -> str | None:
    """
    Run 4-step narrowing classification using predefined definition groups.
    Returns the final selected label or None if the flow breaks at any step.
    """
    # -------------- Load prompt config --------------
    system_prompt = prompt_config["system_prompt"]
    definitions = prompt_config["definitions"]
    groups = prompt_config["groups"]

    selected = []

    # -------------- Step 1–3: Predict one label from each group --------------
    for step, group in enumerate(groups[:3], 1):
        prediction = _predict_from_group(prompter, user_input, system_prompt, group, definitions)

        if prediction not in group:
            print(f"[!] Step {step}: Invalid prediction '{prediction}' not in group {group}")
            return 'Misclassified'

        selected.append(prediction)

    # -------------- Step 4: Final decision from previous selections + 1 remaining --------------
    remaining = [label for label in groups[3] if label not in selected]
    final_group = selected + (remaining[:1] if remaining else [])

    prediction = _predict_from_group(prompter, user_input, system_prompt, final_group, definitions)

    if prediction not in final_group:
        print(f"[!] Final step: Invalid prediction '{prediction}' not in final group {final_group}")
        return 'Misclassified'

    return prediction


def _predict_from_group(prompter, user_input: str, system_prompt: str, group: list[str], definitions: dict) -> str:
    """
    Build prompt from definitions, send to model, return uppercase prediction.
    """
    # -------------- Format prompt with definitions for this group --------------
    descs = {label: definitions[label] for label in group}
    prompt_text = build_group_prompt(user_input, descs)

    # -------------- Build message sequence and query Ollama model --------------
    messages = prompter.build_custom_prompt(system_prompt, prompt_text, {"text": ""})
    return prompter.send_prompt_to_model(messages).strip().upper()

def build_group_prompt(user_input: str, label_definitions: dict) -> str:
    """
    Builds a prompt using square-bracketed input and a list of category definitions.
    """
    label_list = "\n".join([f"{label}: {desc}" for label, desc in label_definitions.items()])
    return f"[{user_input}]\n\n{label_list}"

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

        # Format safety-specific prompt if the task is "safety"
        if cfg.task == 'safety':
            input_text = format_safety_prompt(input_text)

        # Decide which prompting strategy to use
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