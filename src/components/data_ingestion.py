import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started running data ingestion")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Done reading dataset as a dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Started initiating train and test split")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=13)
            
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion has been completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()
#     train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
#     model_r2_score = ModelTrainer().initiate_model_trainer(train_array=train_arr, test_array=test_arr)
    
#     print("Best model r2 score: ", model_r2_score)