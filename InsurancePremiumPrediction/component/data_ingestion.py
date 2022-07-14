
from InsurancePremiumPrediction.logger import logging
from InsurancePremiumPrediction.Exception import InsuranceException
from InsurancePremiumPrediction.config import configuration
from InsurancePremiumPrediction.entity.config_entity import DataIngestionConfig
from InsurancePremiumPrediction.entity.artifact_entity import DataIngestionArtifact
from InsurancePremiumPrediction.constant import *
import os ,sys

from six.moves import urllib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:
        try:
            logging.info(f"{'+'*20} Data Ingestion Log Started .{'+'*20}")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def download_insurance_data(self,):
        try:
            download_url=self.data_ingestion_config.dataset_download_url

            tgz_download_url=self.data_ingestion_config.tgz_download_dir

            if os.path.exists(tgz_download_url):
                os.remove(tgz_download_url)
            os.makedirs(tgz_download_url,exist_ok=True)

            insurance_file_name=DATA_INGESTION_TGZ_DOWNLOAD_FILE_NAME_KEY
            tgz_file_path=os.path.join(tgz_download_url,insurance_file_name)
            logging.info(f"Downloading file from :[{download_url}]")
            urllib.request.urlretrieve(download_url,tgz_file_path)
            logging.info(f"File [{download_url}] has been downloaded successfully at [{tgz_file_path}] . ")
            
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def extract_tgz_file(self,tgz_file_path:str):
        try:
            pass
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def split_data_as_train_test(self,)-> DataIngestionArtifact:
        try:
            pass
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def initiate_data_ingestion(Self,)->DataIngestionArtifact:
        try:
            pass
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def __del__(self):
        logging.info(f"{'#'*20} Data Ingestion Log Completed .{'#'*20} \n\n")


    