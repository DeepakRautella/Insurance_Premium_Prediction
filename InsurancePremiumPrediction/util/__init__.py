import yaml
import os,sys
from InsurancePremiumPrediction.Exception import InsuranceException
from InsurancePremiumPrediction.constant import *

def read_yaml_file(file_path:str)->dict:
    """
    Read yaml file and return the content as a dictionary .
    file_path:str
    """
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise InsuranceException(e,sys)  from e 
        