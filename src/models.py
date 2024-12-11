from abc import ABC, abstractmethod
from typing import Literal, Optional, Any

from pydantic import BaseModel
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

"""
ClassificationLoader(ABC):
    # must contain two columns - "text" , "label"
    # must take labels which is a map between int_label -> str_label
    # Supervised training.
    
    def preprocess(self):
        # assert that the labels are valid.
        # drop rows that have invalid or missing information.
        # map labels to their string form.
    
    # preprocess must update the label column if it's an int.
    pass

"""


class BaseLoader(ABC):
    """Base Loader class used for loading and preprocessing datasets"""

    @abstractmethod
    def load(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def split(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> bool:
        pass

    @property
    @abstractmethod
    def train_data(self) -> Any:
        raise NotImplementedError("Abstract method")

    @property
    @abstractmethod
    def test_data(self) -> Any:
        raise NotImplementedError("Abstract method")

    @property
    @abstractmethod
    def validation_data(self) -> Any:
        raise NotImplementedError("Abstract method")


class BaseInjector(ABC):
    # The injector will be resposible for creating synthetic private data and injecting it in the dataset we are using
    pass


class TransformerConfig(BaseModel):

    device_map: Literal["auto", "balanced", "sequential", "custom", "none"]
    torch_dtype: Literal["float16", "float32", "bfloat16", "int8"]
    quantization_config: Optional[BitsAndBytesConfig] = None


class BaseTransformer(ABC):
    def __init__(self):
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def named_modules(self):
        """Returns the names of all modules in the model used by Lora Training"""
        return self.model.named_modules()

    pass


class BaseTrainer(ABC):
    @abstractmethod
    @property
    def modules(self) -> list[Any]:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def load(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def save(self, output_dir: str, *args, **kwargs) -> bool:
        raise NotImplementedError("Abstract method")


class BaseBenchmarker(ABC):
    # The benchmarker will evaluate the results of the
    pass


class FinetunerConfig(ABC):
    # dataset path
    # model name
    # *args
    # **kwargs
    pass


class BaseFinetuner(ABC):
    def __init__(
        self,
        loader: BaseLoader,
        injector: BaseInjector,
        trainer: BaseTrainer,
        benchmarker: BaseBenchmarker,
    ) -> None:
        self.loader: BaseLoader = loader
        self.injector: BaseInjector = injector
        self.trainer: BaseTrainer = trainer
        self.benchmarker: BaseBenchmarker = benchmarker

    def run(self):
        pass


class QConfig:
    pass
