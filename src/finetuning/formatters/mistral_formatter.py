from finetuning.base_trainer import BasePromptFormatter
import re


# ──────────────────────────────────────────────────────────────
# Mistral Safety Prompt Formatter
# ──────────────────────────────────────────────────────────────

class MistralSafetyPromptFormatter(BasePromptFormatter):
    def __init__(self):
        self.label_map = {"Y": "Safe", "N": "Unsafe"}

    def format_prompt_training(self, struggle, label):
        """
        Formats input for training (safety classification).
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response_text = match.group(3).strip()
        readable_label = self.label_map.get(label, label)

        prompt = f"""[INST] You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations. [/INST]
Struggle: {struggle_text}
Response: {response_text}
Classification: {readable_label}
"""
        return {"text": prompt}

    def format_prompt_inference(self, struggle):
        """
        Formats input for inference (safety classification).
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response_text = match.group(3).strip()

        prompt = f"""[INST] You evaluate doctor responses to patient struggles and classify them as 'Safe' or 'Unsafe'. Provide only the classification label without explanations. [/INST]
Struggle: {struggle_text}
Response: {response_text}
Classification:
"""
        return prompt


# ──────────────────────────────────────────────────────────────
# Mistral Topic Prompt Formatter
# ──────────────────────────────────────────────────────────────

class MistralTopicPromptFormatter(BasePromptFormatter):
    def format_prompt_training(self, struggle, label):
        """
        Formats input for training (topic classification).
        """
        prompt = f"""[INST] Categorize the patient's struggle into one of the following topics:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".
Provide only the topic label without explanation. [/INST]
Struggle: {struggle}
Classification: {label}
"""
        return {"text": prompt}

    def format_prompt_inference(self, struggle):
        """
        Formats input for inference (topic classification).
        """
        prompt = f"""[INST] Categorize the patient's struggle into one of the following topics:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".
Provide only the topic label without explanation. [/INST]
Struggle: {struggle}
Classification:
"""
        return prompt
