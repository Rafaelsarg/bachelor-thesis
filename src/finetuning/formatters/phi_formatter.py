from finetuning.base_trainer import BasePromptFormatter
import re


# ──────────────────────────────────────────────────────────────
# Phi Safety Prompt Formatter Casual Language Model
# ──────────────────────────────────────────────────────────────

class PhiSafetyPromptFormatterCasual(BasePromptFormatter):
    def __init__(self):
        self.label_map = {"Y": "Safe", "N": "Unsafe"}

    def format_prompt_training(self, struggle, label):
        """
        Formats training input for safety tasks.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response = match.group(3).strip()
        readable_label = self.label_map.get(label, label)

        phi_prompt = f"""<|system|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|end|>
<|user|>
Struggle: {struggle_text}
Response: {response}<|end|>
<|assistant|>
Classification: {readable_label}<|end|>"""
        return {"text": phi_prompt}

    def format_prompt_inference(self, struggle):
        """
        Formats inference input for safety tasks.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response = match.group(3).strip()

        phi_prompt = f"""<|system|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|end|>
<|user|>
Struggle: {struggle_text}
Response: {response}<|end|>
<|assistant|>
Classification:
"""
        return phi_prompt

# ──────────────────────────────────────────────────────────────
# Phi Safety Prompt Formatter Classification Language Model
# ──────────────────────────────────────────────────────────────

class PhiSafetyPromptFormatterClassification(BasePromptFormatter):
    def __init__(self):
        self.label_id_map = {"Y": 1, "N": 0}

    def format_prompt_training(self, struggle, label):
        """
        Prepares input for sequence classification using Phi-style prompts.
        Does not include the target label in the text prompt.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response = match.group(3).strip()
        label_id = self.label_id_map[label]

        phi_prompt = f"""<|system|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|end|>
<|user|>
Struggle: {struggle_text}
Response: {response}<|end|>
<|assistant|>
"""

        return {"text": phi_prompt, "label": label_id}

    def format_prompt_inference(self, struggle):
        """
        Formats input for inference (Phi-style chat prompt), excluding label.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response = match.group(3).strip()

        phi_prompt = f"""<|system|>
You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations.<|end|>
<|user|>
Struggle: {struggle_text}
Response: {response}<|end|>
<|assistant|>
"""
        return phi_prompt
        
# ──────────────────────────────────────────────────────────────
# Phi Topic Prompt Formatter
# ──────────────────────────────────────────────────────────────

class PhiTopicPromptFormatter(BasePromptFormatter):
    def format_prompt_training(self, struggle, label):
        """
        Formats training input for topic tasks.
        """
        prompt = f"""<|system|>
Categorize the patient's struggle into one of the following topics:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".
Provide only the topic label without explanations.<|end|>
<|user|>
Struggle: {struggle}<|end|>
<|assistant|>
Classification: {label}<|end|>"""
        return {"text": prompt}

    def format_prompt_inference(self, struggle):
        """
        Formats inference input for topic tasks.
        """
        prompt = f"""<|system|>
Categorize the patient's struggle into one of the following topics:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".
Provide only the topic label without explanations.<|end|>
<|user|>
Struggle: {struggle}<|end|>
<|assistant|>
Classification:
"""
        return prompt
