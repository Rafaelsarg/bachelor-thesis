from finetuning.base_trainer import BasePromptFormatter
import re


# ──────────────────────────────────────────────────────────────
# Llama Safety Prompt Formatter Causal Language Model
# ──────────────────────────────────────────────────────────────

class LlamaSafetyPromptFormatterCausal(BasePromptFormatter):
    def __init__(self):
        self.label_map = {"Y": "Safe", "N": "Unsafe"}

    def format_prompt_training(self, struggle, label):
        """
        Converts patient struggle and doctor response into LLaMA's structured format.

        Args:
            struggle (str): The input text containing patient struggles and doctor responses.
            label (str): Classification label ('Y' or 'N').

        Returns:
            dict: A dictionary containing formatted input and output.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response = match.group(3).strip()
        readable_label = self.label_map.get(label, label)

        llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
Struggle: {struggle_text}
Response: {response}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Classification: {readable_label}<|eot_id|>"""

        return {"text": llama_prompt}

    def format_prompt_inference(self, struggle):
        """
        Formats input for inference. Includes only struggle and response, no label.

        Args:
            struggle (str): The input text containing patient struggles and doctor responses.

        Returns:
            str: A formatted string prompt for model inference.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response_text = match.group(3).strip()

        llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
Struggle: {struggle_text}
Response: {response_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Classification:"""

        return llama_prompt


# ──────────────────────────────────────────────────────────────
# Llama Safety Prompt Formatter Classification Language Model
# ──────────────────────────────────────────────────────────────
class LlamaSafetyPromptFormatterClassification(BasePromptFormatter):
    def __init__(self):
        self.label_id_map = {"Y": "Safe", "N": "Unsafe"}

    def format_prompt_training(self, struggle, label):
        """
        Formats input for sequence classification (LLaMA style).
        Removes label from text, returns it as integer.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response = match.group(3).strip()
        label_id = self.label_id_map[label]

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
Struggle: {struggle_text}
Response: {response}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        return {"text": prompt, "label": label_id}

    def format_prompt_inference(self, struggle):
        """
        Formats input for inference without label, using LLaMA chat-style prompt.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response = match.group(3).strip()

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
Struggle: {struggle_text}
Response: {response}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# ──────────────────────────────────────────────────────────────
# Llama Topic Prompt Formatter
# ──────────────────────────────────────────────────────────────

class LlamaTopicPromptFormatterCausal(BasePromptFormatter):
    def format_prompt_training(self, struggle, label):
        """
        For Causal LM: Generate a prompt and return 'text', which will be both input and target.

        Args:
            struggle (str): The patient's struggle.
            label (str): One of 12 topic classes.

        Returns:
            dict: {"text": prompt}
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Categorize the patient's struggle into **one** of the following categories:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Struggle: {struggle}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Classification: {label}<|eot_id|>"""
        return {"text": prompt}

    def format_prompt_inference(self, struggle):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Categorize the patient's struggle into **one** of the following categories:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Struggle: {struggle}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

# ──────────────────────────────────────────────────────────────
# Llama Topic Prompt Formatter
# ──────────────────────────────────────────────────────────────

class LlamaTopicPromptFormatterClassification(BasePromptFormatter):
    def format_prompt_training(self, struggle, label):
        """
        For Sequence Classification: Return 'text' and numeric 'label'.

        Args:
            struggle (str): The struggle text.
            label (str): String label (e.g., "SOCIAL").

        Returns:
            dict: {"text": prompt, "labels": int}
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Categorize the patient's struggle into **one** of the following categories:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Struggle: {struggle}<|eot_id|>"""

        return {
            "text": prompt,
            "label": label
        }

    def format_prompt_inference(self, struggle):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Categorize the patient's struggle into **one** of the following categories:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Struggle: {struggle}<|eot_id|>"""
