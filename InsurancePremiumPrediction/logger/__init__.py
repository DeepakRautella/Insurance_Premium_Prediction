

import logging
from InsurancePremiumPrediction.constant import *
import os,sys
from stat import filemode

LOG_DIR="logs"

def get_log_file_name():
    return f"log_{get_current_time_stamp()}.log"

LOG_FILE_NAME=get_log_file_name()
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_DIR,LOG_FILE_NAME)


logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
format='[%(asctime)s]-%(name)s-%(levelname)s-%(message)s',
level=logging.INFO
)