import ollama
from typing import Dict, List, Optional


class OllamaPrompting:
    """Wrapper for constructing and sending structured prompts to Ollama models.

    Supports zero-shot, few-shot, and custom prompting strategies for text classification or generation.
    """
    def __init__(self, model_name: str, system_instruction: str):
        """Initialize the prompting instance with an Ollama model and system instruction.

        Args:
            model_name (str): Identifier of the Ollama model (e.g., 'mistral:7b').
            system_instruction (str): Default system-level instruction for the model.
        """
        self.model_name = model_name
        self.system_instruction = system_instruction

    def _create_base_system_message(self) -> List[Dict[str, str]]:
        """Create a base system message for prompt construction.

        Returns:
            List[Dict[str, str]]: List with a single system message dictionary.
        """
        return [{"role": "system", "content": self.system_instruction}]

    def send_prompt_to_model(self, message_sequence: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Send structured messages to the Ollama model and retrieve the response.

        Args:
            message_sequence (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
            temperature (float): Sampling temperature (0.0 for deterministic, 1.0 for creative).

        Returns:
            str: Model's response content.
        """
        response = ollama.chat(
            model=self.model_name,
            messages=message_sequence,
            options={"temperature": temperature}
        )
        return response["message"]["content"].strip()

    def build_zero_shot_prompt(self, user_input_text: str) -> List[Dict[str, str]]:
        """Construct a zero-shot prompt with system instruction and user input.

        Args:
            user_input_text (str): Input text or query for the model.

        Returns:
            List[Dict[str, str]]: Structured message list for zero-shot prompting.
        """
        messages = self._create_base_system_message()
        messages.append({"role": "user", "content": user_input_text})
        return messages

    def build_few_shot_prompt(self, user_input_text: str, example_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Construct a few-shot prompt with examples and user input.

        Args:
            user_input_text (str): Input text or query to process.
            example_pairs (List[Dict[str, str]]): List of example dictionaries with 'input' and 'output' keys.

        Returns:
            List[Dict[str, str]]: Structured message list with examples and query.

        Raises:
            ValueError: If example_pairs is empty.
        """
        if not example_pairs:
            raise ValueError("Few-shot prompting requires at least one example.")

        messages = self._create_base_system_message()
        for example in example_pairs:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
        messages.append({"role": "user", "content": user_input_text})
        return messages

    def build_custom_prompt(self, custom_system_instruction: str, user_input_text: str, context_example: Dict[str, str]) -> List[Dict[str, str]]:
        """Construct a custom prompt with a specific system instruction and example.

        Args:
            custom_system_instruction (str): Custom system-level instruction for the model.
            user_input_text (str): Input text or query to process.
            context_example (Dict[str, str]): Dictionary with a 'text' key for context.

        Returns:
            List[Dict[str, str]]: Structured message list for custom prompting.
        """
        messages = [{"role": "system", "content": custom_system_instruction}]
        messages.append({"role": "user", "content": context_example["text"]})
        messages.append({"role": "user", "content": user_input_text})
        return messages


# ---------------- Custom Prompting Functions ----------------
def run_custom_prompt(prompter, user_input: str, prompt_config: dict) -> Optional[str]:
    """Execute a 4-step narrowing classification using predefined label groups.

    Iteratively predicts labels from groups, building a final group from prior selections and one remaining label.

    Args:
        prompter: Instance of OllamaPrompting for model interaction.
        user_input (str): Input text to classify.
        prompt_config (dict): Configuration with system prompt, definitions, and label groups.

    Returns:
        Optional[str]: Final predicted label or 'Misclassified' if the process fails.
    """
    # Extract prompt configuration
    system_prompt = prompt_config["system_prompt"]
    definitions = prompt_config["definitions"]
    groups = prompt_config["groups"]

    selected = []

    # Steps 1-3: Predict one label from each group
    for step, group in enumerate(groups[:3], 1):
        prediction = _predict_from_group(prompter, user_input, system_prompt, group, definitions)
        if prediction not in group:
            print(f"[!] Step {step}: Invalid prediction '{prediction}' not in group {group}")
            return 'Misclassified'
        selected.append(prediction)

    # Step 4: Final prediction from selected labels and one remaining label
    remaining = [label for label in groups[3] if label not in selected]
    final_group = selected + (remaining[:1] if remaining else [])
    prediction = _predict_from_group(prompter, user_input, system_prompt, final_group, definitions)

    if prediction not in final_group:
        print(f"[!] Final step: Invalid prediction '{prediction}' not in final group {final_group}")
        return 'Misclassified'

    return prediction


def _predict_from_group(prompter, user_input: str, system_prompt: str, group: List[str], definitions: Dict[str, str]) -> str:
    """Generate a prompt for a group of labels and predict using the model.

    Args:
        prompter: Instance of OllamaPrompting for model interaction.
        user_input (str): Input text to classify.
        system_prompt (str): System instruction for the model.
        group (List[str]): List of labels in the current group.
        definitions (Dict[str, str]): Dictionary mapping labels to their descriptions.

    Returns:
        str: Predicted label in uppercase.
    """
    # Format prompt with label definitions
    descs = {label: definitions[label] for label in group}
    prompt_text = build_group_prompt(user_input, descs)

    # Send prompt to model
    messages = prompter.build_custom_prompt(system_prompt, prompt_text, {"text": ""})
    return prompter.send_prompt_to_model(messages).strip().upper()


def build_group_prompt(user_input: str, label_definitions: Dict[str, str]) -> str:
    """Construct a prompt with user input and label definitions.

    Args:
        user_input (str): Input text to classify.
        label_definitions (Dict[str, str]): Dictionary of labels and their descriptions.

    Returns:
        str: Formatted prompt string with input and label definitions.
    """
    label_list = "\n".join([f"{label}: {desc}" for label, desc in label_definitions.items()])
    return f"[{user_input}]\n\n{label_list}"