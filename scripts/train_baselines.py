import hydra
from omegaconf import DictConfig, OmegaConf

# ---- LOCAL IMPORTS ---
from src.baselines.tasks.safety_trainer import SafetyClassificationTrainer
from src.baselines.tasks.cluster_trainer import ClusterClassificationTrainer

# ------------------- Main Function -------------------
@hydra.main(config_path="../../configs", config_name="baseline_config", version_base=None)
def main(cfg: DictConfig):
    # -------- LOAD CONFIGURATION --------
    print("Loaded Config:")
    print(OmegaConf.to_yaml(cfg))

    vec_type = cfg.vectorizer.type                # Extract vectorizer type
    clf_type = cfg.classifier.type                # Extract classifier type

    # -------- PREPARE GRID SEARCH PARAMETERS --------
    # Vectorizer params: use grid search only for bow and tfidf
    vec_params = cfg.vectorizer.grid_params if vec_type in ["bow", "tfidf"] else None

    # Classifier params: select grid search params based on chosen classifier
    clf_params = cfg.classifier.grid_params[clf_type]

    # -------- SELECT TRAINER BASED ON TASK TYPE --------
    if cfg.type == "safety":
        trainer = SafetyClassificationTrainer(
            dataset_path=cfg.dataset_path,
            vectorizer_type=vec_type,
            vectorizer_params=vec_params,
            classifier_type=clf_type,
            classifier_params=clf_params
        )
    elif cfg.type == "cluster":
        trainer = ClusterClassificationTrainer(
            dataset_path=cfg.dataset_path,
            vectorizer_type=vec_type,
            vectorizer_params=vec_params,
            classifier_type=clf_type,
            classifier_params=clf_params
        )
    else:
        raise ValueError(f"Unsupported training type: {cfg.type}")

    # -------- PROCESS THE DATA --------
    print(f"Processing data for type: {cfg.type}, vectorizer: {vec_type}, classifier: {clf_type}")
    X_train, X_test, y_train, y_test = trainer.preprocess_data()

    # -------- RUN PIPELINE --------
    print("Running training pipeline...")
    trainer.run_pipeline(X_train, X_test, y_train, y_test)

# ------------------- Entry Point -------------------
if __name__ == "__main__":
    main()
