import json
import os
from typing import Literal

from black import Optional
from peft import LoraConfig
from pydantic import BaseModel
from transformers import BitsAndBytesConfig, TrainingArguments


class LoaderSplitConfig(BaseModel):
    train: str
    test: Optional[str] = None
    validate: Optional[str] = None


class LoaderConfig(BaseModel):
    url: str
    split: LoaderSplitConfig
    categories_map: dict[int, str]


class TransformerQConfig(BaseModel):
    device_map: Literal["auto", "balanced", "sequential", "custom", "none"] = "auto"
    torch_dtype: Literal["float16", "float32", "bfloat16", "int8"] = "float16"
    quantization_config: Optional[BitsAndBytesConfig] = None


class TransformerConfig(BaseModel):
    base_model_name: str
    config: TransformerQConfig


class QLoraTrainerConfig(BaseModel):
    """Configuration for the Lora Trainer. Used it to do shit"""

    lora_config: LoraConfig
    training_args: TrainingArguments


class FinetuneConfig(BaseModel):
    dataset: LoaderConfig
    transformer: TransformerConfig
    trainer: QLoraTrainerConfig


def load_config(config_path: str) -> FinetuneConfig:
    assert os.path.exists(config_path), "Path does not exist"
    with open(config_path, "r") as f:
        config: FinetuneConfig = FinetuneConfig.model_validate(json.loads(f.read()))
        return config
