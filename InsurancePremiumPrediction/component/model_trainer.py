
from InsurancePremiumPrediction.logger import logging
from InsurancePremiumPrediction.Exception import InsuranceException
from InsurancePremiumPrediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from InsurancePremiumPrediction.entity.config_entity import ModelTrainerConfig
from InsurancePremiumPrediction.util.util import load_numpy_array_data, load_object,save_object
import os,sys
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._regression import r2_score,mean_squared_error
import pickle
import numpy as np


class TrainingTesting:

    def training_models(x_train,y_train,x_test,y_test):
        try:
            model_dict={}
            linear_model=LinearRegression()
            decision_tree_model=DecisionTreeRegressor()
            random_forest=RandomForestRegressor()
            gradient_booster=GradientBoostingRegressor()

            model_dict['linear_model']=linear_model
            model_dict['decision_tree_model']=decision_tree_model
            model_dict['random_forest']=random_forest
            model_dict['gradient_booster']=gradient_booster

            model_score_set={}
            for i in model_dict.values():
                i.fit(x_train,y_train)
                Y_predict=i.predict(x_test)
                model_score_set[i]=[i.score(x_train,y_train),r2_score(y_test,Y_predict)]
            return model_dict,model_score_set
        except Exception as e:
            raise InsuranceException(e,sys) from e
    
    def get_best_model(self,x_train,y_train,x_test,y_test,base_accuracy=0.7) :
        try:
            logging.info(f"Started Initilizing model training")
            model_dict,model_score_set=TrainingTesting.training_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
            score_set=list(model_score_set.values())
            score_set_array=np.array(score_set)
            max=0
            index=0
            for i in range(4):
                if max <score_set_array[i][1]:
                    max=score_set_array[i][1]
                    index=i
            if max < base_accuracy:
                logging.info(f"FAILED TO GET R2 SCORE ABOVE :{ base_accuracy}")
                return False
            param_grid,model=self.Gridsearch_param_model(index,model_dict)
            best_parameters=self.get_best_param(param_grid,model,x_train,y_train)
            self.grid_post_best_model=self.get_grid_post_model(best_parameters,model,x_train,y_train,x_test,y_test)
            return self.grid_post_best_model
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def Gridsearch_param_model(self,index,model_dict):
        try:
            
            logging.info(f"GridSearch starting on Best model r2 score")
            if index==0:
                self.param_grid={
                                        
                            }
                self.model=model_dict['linear_model']
            if index==1:
                self.param_grid={
                            'max_depth':[2,3,4],
                            'min_samples_split':[4,5,6],
                            'random_state': range(100,500,50)        
                            }
                model=model_dict['decision_tree_model']
            if index==2:
                self.param_grid={
                            'n_estimators':[30,50],
                            'max_depth':[2,3,4],
                            'min_samples_split':[4,5,6],
                            'random_state': range(100,500,50)           
                            }
                self.model=model_dict['random_forest']

            if index==3:
                self.param_grid={
                            'min_samples_leaf':[3,7],
                            'max_depth': [1,2,3],
                            'min_samples_split':[3,4],
                            'random_state': range(100,500,100),
                            'alpha': [0.1,0.9],            
                            }
                self.model=model_dict['gradient_booster']

            return self.param_grid,self.model
        except Exception as e:
            raise InsuranceException(e,sys) from e
    
    def get_best_param(self,param_grid,model,x_train,y_train):
        try:
            grid_cv=GridSearchCV(estimator=model,param_grid=param_grid,cv=4,n_jobs=-1)
            grid_cv.fit(x_train,y_train)
            self.best_param=grid_cv.best_params_
            logging.info(f" Best Parameters for :{ model} is :[{self.best_param }]")
            return self.best_param
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def get_grid_post_model(self,best_param,model,x_train,y_train,x_test,y_test):
        try:
            self.grid_post_model=model.set_params(**best_param)
            self.grid_post_model.fit(x_train,y_train)
            self.grid_post_model.predict(x_test)
            grid_post_test_score=self.grid_post_model.score(x_test,y_test)
            logging.info(f"{grid_post_test_score} r2 score after applying best parameters :")
            return self.grid_post_model

        except Exception as e:
            raise InsuranceException(e,sys) from e


def Model_Accuracy_finder(model_obj,x_train,y_train,x_test,y_test):
    try:
        y_train_predict=model_obj.predict(x_train)
        train_score=r2_score(y_train,y_train_predict)
        y_test_predict=model_obj.predict(x_test)
        test_score=r2_score(y_test,y_test_predict)

        train_rmse=np.sqrt(mean_squared_error(y_train,y_train_predict))
        test_rmse=np.sqrt(mean_squared_error(y_test,y_test_predict))

        model_score=(2* (train_score*test_score))/(train_score+test_score)
            
        return train_rmse,test_rmse,train_score,test_score,model_score
    except Exception as e:
        raise InsuranceException(e,sys) from e 

class InsuranceEstimatorModel:
    
    def __init__(self,preprocessing_object,trained_model_object) -> None:
        try:
            self.preprocessing_object=preprocessing_object
            self.trained_model_object=trained_model_object
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def predict(self,x_prediction):
        try:
            transformed_feature=self.preprocessing_object.transform(x_prediction)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) :
        return f"{type(self.trained_model_object).__name__}()"

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
            logging.info(f"{'>>'*10} Model Trainer Log Started {'<<'*10}")
            
            logging.info(f"Loading transformed training dataset .")
            transformed_train_file_path=self.data_transformation_artifact.transformed_train_file_path
            train_array=load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info(f"Loading Transformed test dataset .")
            transformed_test_file_path=self.data_transformation_artifact.transformed_test_file_path
            test_array=load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info(f"Splitting training and  testing Input and Target features ")
            x_train,y_train,x_test,y_test=train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            
            base_accuracy=self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy :{ base_accuracy }")

            logging.info(f"Inititing Model Selection Operation")
            training_testing=TrainingTesting()
            grid_best_model=training_testing.get_best_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,base_accuracy=base_accuracy)   
            
            preprocessing_obj=load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            trained_model_file_path=self.model_trainer_config.trained_model_file_path

            Insurance_model=InsuranceEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=grid_best_model)
            logging.info(f"saving model at path : { trained_model_file_path }")
            save_object(file_path=trained_model_file_path,obj=Insurance_model)
            
            train_rmse,test_rmse,train_accuracy,test_accuracy,model_accuracy=Model_Accuracy_finder(grid_best_model,x_train,y_train,x_test,y_test)
            model_trainer_artifact=ModelTrainerArtifact(is_trained=True,
                                                        message="Model Trained Successfully : ",
                                                        trained_model_file_path=trained_model_file_path,
                                                        train_rmse=train_rmse,
                                                        test_rmse=test_rmse,
                                                        train_accuracy=train_accuracy,
                                                        test_accuracy=test_accuracy,
                                                        model_accuracy=model_accuracy)
            return model_trainer_artifact
        except Exception as e:
            raise InsuranceException(e,sys) from e
