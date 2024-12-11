import os
from typing import Optional

import bitsandbytes as bnb
import wandb
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

from src.config import QLoraTrainerConfig
from src.models import BaseTrainer, BaseTransformer, BaseLoader


class QLoraTrainer(BaseTrainer):
    def __init__(
        self,
        transformer: BaseTransformer,
        loader: BaseLoader,
        config: QLoraTrainerConfig,
    ) -> None:
        self._transformer: BaseTransformer = transformer
        self._loader: BaseLoader = loader

        self._lora_config: Optional[LoraConfig] = None
        self._training_args: Optional[TrainingArguments] = None
        self._trainer: Optional[SFTTrainer] = None
        self._config: QLoraTrainerConfig = config

        self.load(config=config)

    @property
    def modules(self) -> list[str]:
        output_dir: str = "Phi-3.5-mini-instruct"
        cls = bnb.nn.Linear4bit
        lora_module_names: set = set()
        for name, module in self._transformer.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if "lm_head" in lora_module_names:  # needed for 16 bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    def load(self, config: QLoraTrainerConfig) -> bool:
        """Loading the Lora Trainer"""

        self._config: QLoraTrainerConfig = config
        self._lora_config: Optional[LoraConfig] = config.lora_config
        self._training_args: Optional[TrainingArguments] = (
            QLoraTrainerConfig.training_args
        )
        self._trainer: SFTTrainer = SFTTrainer(
            model=self._transformer.model,
            args=self._training_args,
            train_dataset=self._loader.train_data,
            eval_dataset=self._loader.validation_data,
            peft_config=self._lora_config,
            dataset_text_field="text",
            tokenizer=self._transformer.tokenizer,
            max_seq_length=512,
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
        )
        return True

    def train(
        self, project_name: str, job_type: str = "training", anonymous: str = "allow"
    ) -> None:
        wandb.login(key=os.getenv("WAND_TOKEN"))
        run = wandb.init(project=project_name, job_type=job_type, anonymous=anonymous)
        self._trainer.train()

    def save(self, output_dir: str, *args, **kwargs) -> bool:
        base_dir: str = os.path.dirname(output_dir)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        self._trainer.save_model(output_dir)
        self._trainer.tokenizer.save_pretrained(output_dir)

        return True
