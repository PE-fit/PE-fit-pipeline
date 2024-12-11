import os
from src.config import FinetuneConfig, load_config
from argparse import ArgumentParser, Namespace
from loguru import logger

from src.loader import ClassificationHuggingFaceLoader
from src.models import BaseLoader, BaseTransformer, BaseTrainer
from src.trainer import QLoraTrainer
from src.transformer import HuggingFaceTransformer

parser: ArgumentParser = ArgumentParser(description="Finetune tool")

parser.add_argument("--config", type=str, help="path to config file")
parser.add_argument("--output_dir", type=str, help="path to output save folder")

args: Namespace = parser.parse_args()

if not os.path.exists(args.config):
    logger.error("Config file doesn't exist")
    exit(1)

if os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
elif os.path.dirname(args.output_dir):
    args.output_dir = os.path.dirname(args.output_dir)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)


def generate_prompt(data_point) -> str:
    return f"""
            Classify the medical transcription text into strictly and exactly one of the following categories: Pain_Management, Chiropractic, Podiatry, Pediatrics_Neonatal, Discharge_Summary, Cosmetic_Plastic_Surgery, Neurology, Endocrinology, Rheumatology, Orthopedic, Dentistry, Allergy_Immunology, Psychiatry_Psychology, Consult_History_and_Physical, Dermatology, Radiology, Speech_Language, Physical_Medicine_Rehab, Sleep_Medicine, Hospice_Palliative_Care, Diets_and_Nutrition, Urology, ENT_Otolaryngology, Gastroenterology, Letters, Surgery, Bariatrics, Ophthalmology, Neurosurgery, Emergency_Room_Reports, Nephrology, Lab_Medicine_Pathology, Office_Notes, Cardiovascular_Pulmonary, SOAP_Chart_Progress_Notes, Autopsy, General_Medicine, IME_QME_Work_Comp, Obstetrics_Gynecology, Hematology_Oncology.
text: {data_point["text"]}
label: {data_point["label"]}""".strip()


logger.info(f"Loading config from {args.config}")
config: FinetuneConfig = load_config(args.config)
logger.success("Config loaded successfully")


loader: BaseLoader = ClassificationHuggingFaceLoader(
    url=config.dataset.url,
    split=config.dataset.split,
    categories_map=config.dataset.categories_map,
    generate_prompt=generate_prompt,
)

transformer: BaseTransformer = HuggingFaceTransformer(
    base_model_name=config.transformer.base_model_name, config=config.transformer.config,
    categories=list(config.dataset.categories_map.values())
)

trainer: QLoraTrainer = QLoraTrainer(
    transformer=transformer, loader=loader, config=config.trainer
)

trainer.train(project_name="Testing Finetuning Phi-3.5-mini-instruct on pipeline v1")
trainer.save(output_dir=args.output_dir)
