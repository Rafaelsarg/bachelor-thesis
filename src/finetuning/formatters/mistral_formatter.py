from finetuning.base_trainer import BasePromptFormatter
import re


# ──────────────────────────────────────────────────────────────
# Mistral Safety Prompt Formatter Causal Language Model
# ──────────────────────────────────────────────────────────────

class MistralSafetyPromptFormatterCausal(BasePromptFormatter):
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
# Mistral Safety Prompt Formatter Classification Language Model
# ──────────────────────────────────────────────────────────────

class MistralSafetyPromptFormatterClassification(BasePromptFormatter):
    def __init__(self):
        self.label_id_map = {"Y": 1, "N": 0}

    def format_prompt_training(self, struggle, label):
        """
        Prepares input for sequence classification with integer label.
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response_text = match.group(3).strip()

        prompt = f"""[INST] Evaluate if the response to the struggle is 'Safe' or 'Unsafe'. [/INST]
Struggle: {struggle_text}
Response: {response_text}
"""
        return {"text": prompt, "label": self.label_id_map[label]}

    def format_prompt_inference(self, struggle):
        """
        Prepares input for inference (without label).
        """
        match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", struggle)
        if not match:
            return None

        struggle_text = match.group(1).strip()
        response_text = match.group(3).strip()

        prompt = f"""[INST] Evaluate if the response to the struggle is 'Safe' or 'Unsafe'. [/INST]
Struggle: {struggle_text}
Response: {response_text}
"""
        return prompt

# ──────────────────────────────────────────────────────────────
# Mistral Topic Prompt Formatter Causal Language Model
# ──────────────────────────────────────────────────────────────

class MistralTopicPromptFormatterCausal(BasePromptFormatter):
    def format_prompt_training(self, struggle, label):
        """
        For Causal LM: Return prompt as single 'text' field; label = input_ids.
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
        return f"""[INST] Categorize the patient's struggle into one of the following topics:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".
Provide only the topic label without explanation. [/INST]
Struggle: {struggle}
Classification:"""


# ──────────────────────────────────────────────────────────────
# Mistral Topic Prompt Formatter Classification Language Model
# ──────────────────────────────────────────────────────────────

class MistralTopicPromptFormatterClassification(BasePromptFormatter):
    def format_prompt_training(self, struggle, label):
        """
        For sequence classification: return input prompt and integer label.
        """
        prompt = f"""[INST] Categorize the patient's struggle into one of the following topics:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".
Provide only the topic label without explanation. [/INST]
Struggle: {struggle}
"""
        return {"text": prompt, "label": label}

    def format_prompt_inference(self, struggle):
        return f"""[INST] Categorize the patient's struggle into one of the following topics:
"DIET_PLAN_ISSUES", "SOCIAL", "SITUATIONAL", "MOTIVATION", "EMOTIONS", "CRAVING_HABIT", "MENTAL_HEALTH",
"ENERGY_EFFORT_CONVENIENCE", "PORTION_CONTROL", "KNOWLEDGE", "HEALTH_CONDITION", "NOT_APPLICABLE".
Provide only the topic label without explanation. [/INST]
Struggle: {struggle}
"""


