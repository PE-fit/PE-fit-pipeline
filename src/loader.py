from copy import deepcopy
from typing import Callable, Optional, TypedDict

import pandas as pd
from datasets import Dataset
from loguru import logger
from sklearn.model_selection import train_test_split

from src.models import BaseLoader

"""
first Classification
class ClassificationHuggingFaceLoader():
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_validation: pd.DataFrame
    
    # Methods
    load() -> bool : loads shit.
    split() -> bool: splits the dataset.
    preprocess() -> bool: preprocesses the dataset.
        
"""


class Split(TypedDict):
    train: str
    test: Optional[str]
    validation: Optional[str]


class ClassificationHuggingFaceLoader(BaseLoader):
    """Loading Hugging face datasets - supports text classification tasks. Assumes dataset has"""

    def __init__(
        self,
        url: str,
        split: Split,
        categories_map: dict[int, str],
        generate_prompt: Callable,
    ) -> None:
        self.url: str = url
        self._split: Split = split
        self.categories_map: dict[int, str] = deepcopy(categories_map)
        self.categories_keys: set[int] = set(self.categories_map.keys())
        self.generate_prompt: Callable = generate_prompt

        # raw df_train
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.df_validation: Optional[pd.DataFrame] = None

        self.load()
        self.preprocess()

    def load(self) -> bool:
        logger.info("Loading Hugging Dataset")
        if not self._split.get("train"):
            logger.error("Train dataset not found")
            return False
        self.df_train = pd.read_parquet(f"{self.url}/{self._split.get('train')}")
        if self._split.get("test"):
            self.df_test = pd.read_parquet(f"{self.url}/{self._split.get('test')}")
        if self._split.get("validation"):
            self.df_validation = pd.read_parquet(
                f"{self.url}/{self._split.get('validation')}"
            )

        return self.split()

    def split(
        self, train_size: float = 0.8, eval_size: float = 0.1, test_size: float = 0.1
    ) -> bool:

        total_size: float = train_size + eval_size + test_size
        assert (
            abs(total_size - 1.0) < 1e-6
        ), "Train, eval, and test sizes must sum to 1.0"
        for size, df_name in zip(
            [train_size, eval_size, test_size], ["train", "eval", "test"]
        ):
            assert 0 < size <= 1, f"Invalid split size for {df_name}"

        if self.df_test is None and self.df_validation is None:

            train_data, temp_data = train_test_split(
                self.df_train, test_size=(eval_size + test_size)
            )
            validation_data, test_data = train_test_split(
                temp_data, test_size=(test_size / (eval_size + test_size))
            )

            self.df_train = train_data
            self.df_validation = validation_data
            self.df_test = test_data

        elif self.df_test is None:
            # Split train into train and test
            train_data, test_data = train_test_split(self.df_train, test_size=test_size)
            self.df_train = train_data
            self.df_test = test_data

        elif self.df_validation is None:
            # Split train into train and validation
            train_data, validation_data = train_test_split(
                self.df_train, test_size=eval_size
            )
            self.df_train = train_data
            self.df_validation = validation_data
        logger.success("Successfully loaded the dataset")
        return True

    def preprocess(self) -> bool:
        """ """
        # assert that "label" and "text" are keys.
        # augment the texts
        logger.info("Preprocessing the dataset")

        for df in (self.df_train, self.df_validation, self.df_test):

            if df is None:
                continue

            assert "label" in df.columns, "'label' column not found in dataframe"
            assert "text" in df.columns, "'text' column not found in dataframe"

            labels: set = set(df["label"].unique().tolist())
            assert labels.issubset(self.categories_keys), (
                "Labels are not a subset of categories. Extra labels are found: "
                f"{labels.difference(self.categories_keys)}"
            )
            # labels
            df["label"] = df["label"].apply(lambda idx: self.categories_map[idx])
            df.loc[:, "text"] = df.apply(self.generate_prompt, axis=1)
        logger.success("Successfully preprocessed the dataset")
        return True

    @property
    def train_data(self) -> Dataset:
        return Dataset.from_pandas(self.df_train[["text"]])

    @property
    def test_data(self) -> Dataset:
        return Dataset.from_pandas(self.df_test[["text"]])

    @property
    def validation_data(self) -> Dataset:
        return Dataset.from_pandas(self.df_validation[["text"]])


def generate_prompt(data_point):
    return f"""
            Classify the medical transcription text into strictly and exactly one of the following categories: Pain_Management, Chiropractic, Podiatry, Pediatrics_Neonatal, Discharge_Summary, Cosmetic_Plastic_Surgery, Neurology, Endocrinology, Rheumatology, Orthopedic, Dentistry, Allergy_Immunology, Psychiatry_Psychology, Consult_History_and_Physical, Dermatology, Radiology, Speech_Language, Physical_Medicine_Rehab, Sleep_Medicine, Hospice_Palliative_Care, Diets_and_Nutrition, Urology, ENT_Otolaryngology, Gastroenterology, Letters, Surgery, Bariatrics, Ophthalmology, Neurosurgery, Emergency_Room_Reports, Nephrology, Lab_Medicine_Pathology, Office_Notes, Cardiovascular_Pulmonary, SOAP_Chart_Progress_Notes, Autopsy, General_Medicine, IME_QME_Work_Comp, Obstetrics_Gynecology, Hematology_Oncology.
text: {data_point["text"]}
label: {data_point["label"]}""".strip()


if __name__ == "__main__":
    from dotenv import load_dotenv

    labels = [
        "Pain_Management",
        "Chiropractic",
        "Podiatry",
        "Pediatrics_Neonatal",
        "Discharge_Summary",
        "Cosmetic_Plastic_Surgery",
        "Neurology",
        "Endocrinology",
        "Rheumatology",
        "Orthopedic",
        "Dentistry",
        "Allergy_Immunology",
        "Psychiatry_Psychology",
        "Consult_History_and_Physical",
        "Dermatology",
        "Radiology",
        "Speech_Language",
        "Physical_Medicine_Rehab",
        "Sleep_Medicine",
        "Hospice_Palliative_Care",
        "Diets_and_Nutrition",
        "Urology",
        "ENT_Otolaryngology",
        "Gastroenterology",
        "Letters",
        "Surgery",
        "Bariatrics",
        "Ophthalmology",
        "Neurosurgery",
        "Emergency_Room_Reports",
        "Nephrology",
        "Lab_Medicine_Pathology",
        "Office_Notes",
        "Cardiovascular_Pulmonary",
        "SOAP_Chart_Progress_Notes",
        "Autopsy",
        "General_Medicine",
        "IME_QME_Work_Comp",
        "Obstetrics_Gynecology",
        "Hematology_Oncology",
    ]
    load_dotenv()
    split: Split = Split(
        train="data/train-00000-of-00001.parquet",
        test="data/test-00000-of-00001.parquet",
        validation=None,
    )

    loader: ClassificationHuggingFaceLoader = ClassificationHuggingFaceLoader(
        url="hf://datasets/rungalileo/medical_transcription_40",
        split=split,
        generate_prompt=generate_prompt,
        categories_map=dict(enumerate(labels)),
    )

    print(loader.train_data["text"][3])
    # print(loader.df_train.shape)
    # print(loader.df_validation.shape)
    # print(loader.df_test.shape)
    #
    # print(loader.df_train.head(n=10))
    # print(loader.df_test.head(n=5))
    # print(loader.df_validation.head(n=5))
