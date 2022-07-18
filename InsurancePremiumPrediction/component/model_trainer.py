
from InsurancePremiumPrediction.logger import logging
from InsurancePremiumPrediction.Exception import InsuranceException
from InsurancePremiumPrediction.entity.artifact_entity import DataIngestionArtifact,\
    DataValidationArtifact,DataTransformationArtifact, ModelTrainerArtifact
from InsurancePremiumPrediction.entity.config_entity import ModelTrainerConfig
from InsurancePremiumPrediction.util.util import load_numpy_array_data
import os,sys

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics._regression import r2_score,mean_squared_error
import pickle

class TrainingTesting:
    
    def __init__(self) -> None:
        pass
    
    def training_dataset(self,x_train,y_train,x_test,y_test):
        model_list=[]
        linear_model=LinearRegression()
        decision_tree_model=DecisionTreeRegressor()
        knn_model=KNeighborsRegressor()
        gradient_booster=GradientBoostingRegressor()
        random_booster=RandomForestRegressor()
        model_list.append([linear_model,decision_tree_model,knn_model,gradient_booster,random_booster])
        for i in model_list():
            i.fit()
        
        return model_list

    def Gridsearch_Cv(self,estimator,):
        pass

class InsuranceEstimatorModel:
    
    def __init__(self) -> None:
        pass

    def predict(self):
        pass

    def __repr__(self) -> str:
        pass

    def __str__(self) :
        pass

class ModelTrainer:

    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                      model_trainer_config:ModelTrainerConfig) -> None:
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise InsuranceException(e,sys) from e

 
    
    def initiate_model_trainer(self)-> ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset .")
            transformed_train_file_path=self.data_transformation_artifact.transformed_train_file_path
            train_array=load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info(f"Loading Transformed test dataset .")
            transformed_test_file_path=self.data_transformation_artifact.transformed_test_file_path
            test_array=load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info(f"Splitting training and  testing Input and Target features ")
            x_train,y_train,x_test,y_test=train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            
            logging.info(f"Extracting model config file path")
            #self model training defining

            base_accuracy=self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy :{ base_accuracy }")

            logging.info(f"Inititing Model Selection Operation")
            """ best_model=
            
            
            is_trained=
            message=
            trained_model_file_path=
            train_rmse=
            test_rmse=
            train_accuracy=
            test_accuracy=
            model_accuracy=
            

            """
            model_trainer_artifact=ModelTrainerArtifact(is_trained=is_trained,
                                                        message=message,
                                                        trained_model_file_path=trained_model_file_path,
                                                        train_rmse=train_rmse,
                                                        test_rmse=test_rmse,
                                                        train_accuracy=train_accuracy,
                                                        test_accuracy=test_accuracy,
                                                        model_accuracy=model_accuracy)
            return model_trainer_artifact
        except Exception as e:
            raise InsuranceException(e,sys) from e
