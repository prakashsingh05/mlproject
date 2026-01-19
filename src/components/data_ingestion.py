import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# 🔥 CORRECT PROJECT ROOT (points to D:\Projects\mlproject)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACTS_DIR, "train.csv")
    test_data_path: str = os.path.join(ARTIFACTS_DIR, "test.csv")
    raw_data_path: str = os.path.join(ARTIFACTS_DIR, "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            # ✅ Correct dataset path
            data_path = os.path.join(
                PROJECT_ROOT, "notebook", "data", "stud.csv"
            )

            df = pd.read_csv(data_path)
            logging.info("Dataset read successfully")

            # Create artifacts directory
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # Train-test split
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# 🚀 FULL TRAINING PIPELINE ENTRY POINT
if __name__ == "__main__":
    logging.info("Starting full training pipeline")

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Training completed successfully. R2 Score: {r2}")
