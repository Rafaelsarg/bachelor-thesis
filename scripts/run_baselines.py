import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# ---- LOCAL IMPORT ---
# === Adjust import path for src/ ===
sys.path.append("src")

from baselines.classification import run_pipeline
from baselines.data_processor import preprocess_data

# ------------------- Hydra Main Entry -------------------
@hydra.main(config_path="../config", config_name="baseline_config", version_base=None)
def main(cfg: DictConfig):
    """
    Unified entry point for running safety or cluster classification using a common trainer pipeline.
    """

    # -------- DISPLAY CONFIGURATION --------
    print("Loaded Configuration")
    
    # -------- CONVERT TO SERIALIZABLE DICT --------
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # -------- EXTRACT CONFIGURATION --------
    task_type = config_dict["task"]
    results_dir = config_dict["results_dir"]
    dataset_path = config_dict["dataset_path"]
    vectorizer_type = config_dict["vectorizer"]["type"]
    classifier_type = config_dict["classifier"]["type"]
    sentence_bert_model = config_dict.get("sentence_bert_model", None)
    word2vec_path = config_dict.get("word2vec_path", None)

    # Grid search parameters
    vectorizer_params = config_dict["vectorizer"]["grid_params"] if vectorizer_type in ["bow", "tfidf"] else None
    classifier_params = config_dict["classifier"]["grid_params"]

    # -------- PREPROCESS DATA --------
    print(f"\n🔧 Preprocessing dataset: {dataset_path}")
    X_train, X_test, y_train, y_test, label_mapping = preprocess_data(
        dataset_path=dataset_path,
        vectorizer_type=vectorizer_type,
        task_type=task_type,
        verbose=True
    )

    # -------- RUN TRAINING PIPELINE --------
    print(f"\n🚀 Training with vectorizer={vectorizer_type}, classifier={classifier_type}")
    run_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_mapping=label_mapping,
        vectorizer_type=vectorizer_type,
        vectorizer_params=vectorizer_params,
        clf_name=classifier_type,
        clf_params=classifier_params,
        results_dir=results_dir,
        sentence_bert_model=sentence_bert_model,
        word2vec_path=word2vec_path,
        task_type=task_type,
        verbose=True
    )

# ------------------- Python Entry Point -------------------
if __name__ == "__main__":
    main()
