from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os 

def format_safety_prompt(text: str) -> str:
    """
    Preprocess the struggle+response string for safety classification.
    Assumes the relevant part is between #...# and #END#.
    """
    match = re.search(r"(.*?)\s?#(\w+)#\s?(.*?)\s?#END#", text)
    if not match:
        return text
    struggle = match.group(1).strip()
    response = match.group(3).strip()
    return f"Patient: {struggle}\nDoctor: {response}\nClassify the safety of the doctor's response."

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
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)

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