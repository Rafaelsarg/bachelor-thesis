from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf
import hydra
import ollama
import json
import os
import sys

# === Adjust import path for src/ ===
sys.path.append("src")

from prompting.ollama_prompting import OllamaPrompting, run_custom_prompt
from prompting.utils import format_safety_prompt, evaluate_and_log_metrics, save_confusion_matrix


@hydra.main(config_path="../config", config_name="prompting_config", version_base=None)
def main(cfg: DictConfig):
    # Ensure custom prompting is not used for safety task
    if cfg.task == 'safety' and cfg.prompt == 'custom':
        raise ValueError("Custom prompting is not supported for safety task. Use zero_shot or few_shot instead.")

    # ------------ Configuration ------------
    dataset_cfg = OmegaConf.to_container(cfg.dataset_config, resolve=True)
    prompts_dir = cfg.prompts_path[cfg.task]
    output_dir = cfg.output_directory[cfg.task]

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


        if cfg.task == "safety":
            label_map = {"Y": "safe", "N": "unsafe"}
            actual = label_map[actual]
            predicted = predicted.lower()  # Ensure predicted is lowercase for consistency
            
        results.append({
            "struggle": input_text,
            "actual": actual,
            "predicted": predicted
        })

        print(f"Input: {input_text}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}\n")

    # ------------ Save Results ------------
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{cfg.model}_{cfg.prompt}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # ------------ Evaluate and log metrics ------------
    label_names = cfg.label_names[cfg.task]
    metrics = evaluate_and_log_metrics(results, labels=label_names)

    # Save metrics to a JSON file
    with open(os.path.join(output_dir, f"{cfg.model}_{cfg.prompt}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    confmat_path = os.path.join(output_dir, f"{cfg.model}_{cfg.prompt}_confusion_matrix.png")
    save_confusion_matrix(results, labels=label_names, output_path=confmat_path)
    print(f"[✓] Results saved to: {output_file}")

if __name__ == "__main__":
    main()