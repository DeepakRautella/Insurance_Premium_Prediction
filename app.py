
from InsurancePremiumPrediction.logger import logging
import sys
from InsurancePremiumPrediction.Exception import InsuranceException
from flask import Flask


app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index( ):
    try:
        logging.info("testing logging and exception")
    except Exception as e:
        raise InsuranceException(e,sys) from e
    return "pipeline establisment under process"   
        


if __name__=="__main__":
    app.run()
