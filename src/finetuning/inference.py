import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from typing import List, Tuple
from finetuning.base_inference import BaseInference


class GenericInference(BaseInference):
    """
    Concrete inference class implementing `BaseInference`.
    Supports both single prediction and dataset-level inference.
    """

    def predict(self, struggle: str) -> str:
        """
        Predicts a single label for the given struggle input.

        Args:
            struggle (str): A formatted string containing struggle and response.

        Returns:
            str: Predicted classification label.
        """
        prompt = self.prompt_formatter.format_prompt_inference(struggle)
        if prompt is None:
            return "Unknown"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            if self.model_type == "casual":
                outputs = self.model.generate(**inputs, max_new_tokens=10)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self._extract_prediction(decoded)

            elif self.model_type == "classification":
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred_id = torch.argmax(logits, dim=-1).item()
                return self.model.config.id2label.get(pred_id, "Unknown")

            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

    def predict_dataset(self, dataset: Dataset) -> Tuple[List[str], List[str]]:
        """
        Predicts labels for all examples in a dataset split.

        Args:
            dataset (Dataset): The Hugging Face `Dataset` split.

        Returns:
            Tuple[List[str], List[str]]: predictions, references
        """
        predictions = []
        references = []

        for example in dataset:
            pred = self.predict(example["struggle"])
            predictions.append(pred)
            references.append(example["label"])

        return predictions, references

    def _extract_prediction(self, decoded_text: str) -> str:
        """
        Extracts the classification label from the model output.

        Args:
            decoded_text (str): Full decoded output.

        Returns:
            str: Extracted prediction.
        """
        if "Classification:" in decoded_text:
            return decoded_text.split("Classification:")[-1].strip().split()[0]
        return "Unknown"
