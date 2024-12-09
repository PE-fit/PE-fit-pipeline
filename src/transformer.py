from copy import deepcopy
from typing import Any, Optional

from loguru import logger
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

from src.models import BaseTransformer, TransformerConfig


class HuggingFaceTransformer(BaseTransformer):
    """Class for transformers - supports classification tasks"""

    def __init__(
        self, base_model_name: str, config: TransformerConfig, categories: list[str]
    ) -> None:
        assert isinstance(categories, list) and all(
            isinstance(category, str) for category in categories
        ), "Categories should be a list of strings"

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_name: str = base_model_name
        self.config: TransformerConfig = config
        self.categories: list[str] = deepcopy(categories)
        assert self.load()
        self.pipe: pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=40,
            temperature=0.1,
        )

    def load(self) -> bool:
        """Loading the pretrained model, tokenizer and other stuff"""
        self.tokenizer: Optional[Any] = AutoTokenizer.from_pretrained(
            self.model_name
        )  # tokenizer for the model to finetune
        logger.info(self.config.model_dump())
        self.model: Optional[Any] = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            quantization_config=self.config.quantization_config,
            device_map=self.config.device_map,
            torch_dtype=self.config.torch_dtype,
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        return True

    def predict(self, prompt: str) -> str:
        """Given prompt and categories , predict where category is the prompt"""
        result = self.pipe(prompt)
        answer = result[0]["generated_text"].split("label:")[-1].strip()

        for category in self.categories:
            if category.lower() in answer.lower():
                return category

        logger.info("Invalid category")
        return self.categories[0]


if __name__ == "__main__":
    base_model_name = "microsoft/Phi-3.5-mini-instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    config: TransformerConfig = TransformerConfig(
        device_map="auto", torch_dtype="float16", quantization_config=bnb_config
    )
    transformer: HuggingFaceTransformer = HuggingFaceTransformer(
        base_model_name=base_model_name, config=config, categories=[""]
    )
