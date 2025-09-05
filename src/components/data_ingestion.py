import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    val_data_path: str=os.path.join('artifacts', "validation.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'notebook\data\spam.csv')
            df.drop_duplicates()
            logging.info('Read the dataset and drop duplicates')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test val split initiated")

            temp, test = train_test_split(
                df, test_size=0.15, random_state=42, stratify=df["Category"])

            train, val = train_test_split(
                temp, test_size=0.176, random_state=42, stratify=temp["Category"])
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            val.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, val_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train_arr, X_test_arr, y_train_arr, y_test_arr, X_val_arr, y_val_arr,_= data_transformation.initiate_data_transformation(train_data, test_data, val_data)
