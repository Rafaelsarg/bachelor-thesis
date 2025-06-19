from finetuning.base_trainer import BasePromptFormatter, BaseTrainer
from transformers import DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
import wandb


# ──────────────────────────────────────────────────────────────
# Generic Trainer
# ──────────────────────────────────────────────────────────────

class GenericTrainer(BaseTrainer):
    def __init__(self, model_name, dataset_path, output_dir, cfg, prompt_formatter: BasePromptFormatter):
        super().__init__(model_name, dataset_path, output_dir, cfg, prompt_formatter)

        self.training_args = SFTConfig(
            output_dir=self.checkpoints_dir, 
            per_device_train_batch_size=self.sft_cfg.batch_size,
            num_train_epochs=self.sft_cfg.epochs,
            learning_rate=self.sft_cfg.lr,
            save_strategy="epoch",
            bf16=True,
            gradient_accumulation_steps=self.sft_cfg.gradient_accumulation_steps,
            max_grad_norm=self.sft_cfg.max_grad_norm,
            lr_scheduler_type=self.sft_cfg.lr_scheduler_type,
            warmup_ratio=self.sft_cfg.warmup_ratio,
            optim=self.sft_cfg.optim,
            logging_dir=self.logs_output_dir,
            logging_steps=self.sft_cfg.logging_steps,
            max_seq_length=self.sft_cfg.max_seq_length,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

    def train(self):
        self.trainer.train()
        print("Training completed.")

        # Save the model in safetensors format
        self.model.save_pretrained(self.model_output_dir, safe_serialization=True)

        print(f"✅ Model and tokenizer saved to: {self.model_output_dir}")
        wandb.finish()
