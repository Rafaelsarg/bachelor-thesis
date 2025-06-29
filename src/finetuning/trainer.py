from finetuning.base_trainer import BasePromptFormatter, BaseTrainer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from torch.utils.data import DataLoader
import wandb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),   # Use "binary" if binary classification
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

# ──────────────────────────────────────────────────────────────
# Generic Trainer
# ──────────────────────────────────────────────────────────────

class GenericTrainer(BaseTrainer):
    def __init__(self, model_name, output_dir, cfg, prompt_formatter: BasePromptFormatter):
        super().__init__(model_name, output_dir, cfg, prompt_formatter)

        if self.model_type == "causal":
            self.training_args = SFTConfig(
                output_dir=self.checkpoints_dir,
                logging_dir=self.logs_output_dir,
                save_strategy="epoch",
                bf16=True,
                per_device_train_batch_size=self.sft_cfg["batch_size"],
                num_train_epochs=self.sft_cfg["epochs"],
                learning_rate=self.sft_cfg["lr"],
                gradient_accumulation_steps=self.sft_cfg["gradient_accumulation_steps"],
                max_grad_norm=self.sft_cfg["max_grad_norm"],
                lr_scheduler_type=self.sft_cfg["lr_scheduler_type"],
                warmup_ratio=self.sft_cfg["warmup_ratio"],
                optim=self.sft_cfg["optim"],
                logging_steps=self.sft_cfg["logging_steps"],
                max_seq_length=self.sft_cfg["max_seq_length"]
            )

            self.trainer = SFTTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["validation"],
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )

        elif self.model_type == "classification":
            self.training_args = SFTConfig(
                output_dir=self.checkpoints_dir,
                logging_dir=self.logs_output_dir,
                save_strategy="epoch",
                eval_strategy="epoch",
                load_best_model_at_end=True,                     # ← track & reload best model
                per_device_train_batch_size=self.sft_cfg["batch_size"],
                num_train_epochs=self.sft_cfg["epochs"],
                learning_rate=self.sft_cfg["lr"],
                gradient_accumulation_steps=self.sft_cfg["gradient_accumulation_steps"],
                max_grad_norm=self.sft_cfg["max_grad_norm"],
                lr_scheduler_type=self.sft_cfg["lr_scheduler_type"],
                warmup_ratio=self.sft_cfg["warmup_ratio"],
                optim=self.sft_cfg["optim"],
                logging_steps=self.sft_cfg["logging_steps"],
                report_to=["wandb"],
                remove_unused_columns=False, 
                bf16=True,
            )

            self.trainer = SFTTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["validation"],
                data_collator=DataCollatorWithPadding(self.tokenizer, return_tensors="pt"),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)],
                compute_metrics=compute_metrics,  
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train(self):
        self.trainer.train()
        print("Training completed.")

        # Save the model in safetensors format
        self.model.save_pretrained(self.model_output_dir, safe_serialization=True)

        print(f"Model and tokenizer saved to: {self.model_output_dir}")
        wandb.finish()
