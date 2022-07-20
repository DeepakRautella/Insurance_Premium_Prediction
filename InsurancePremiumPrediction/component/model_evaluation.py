from InsurancePremiumPrediction.logger import logging
from InsurancePremiumPrediction.Exception import InsuranceException
from InsurancePremiumPrediction.component.model_trainer import Model_Accuracy_finder
from InsurancePremiumPrediction.entity.artifact_entity import ModelEvaluationArtifact, \
    ModelTrainerArtifact,DataTransformationArtifact
from InsurancePremiumPrediction.entity.config_entity import ModelEvaluationConfig
from InsurancePremiumPrediction.util.util import load_numpy_array_data, load_object,\
    read_yaml_file,write_yaml_file
import os,sys
from InsurancePremiumPrediction.constant import *


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def get_best_model(self,):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model
            
            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            logging.info(f"Previous best model loaded successfull { model }")
            return model
        except Exception as e:
            raise InsuranceException(e,sys) from e

    def update_evaluation_report(self,model_evaluation_artifact: ModelEvaluationArtifact):    
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history
                                }

                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise InsuranceException(e,sys) from e

    def initiate_model_evaluation(self,)-> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)
            
            train_array_path=self.data_transformation_artifact.transformed_train_file_path
            test_array_path=self.data_transformation_artifact.transformed_test_file_path
            train_array=load_numpy_array_data(train_array_path)
            test_array=load_numpy_array_data(test_array_path)
            x_train,y_train,x_test,y_test=train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            model = self.get_best_model()
            
            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            
            logging.info(f"{'*'*10}Comparision between Best Model and New Trained Model {'*'*10}")
            
            trained_model_score=self.model_trainer_artifact.model_accuracy
            
            logging.info(f" Previous Best Model : { model }")
            model_obj = model.trained_model_object
            _,_,_,_,model_score=Model_Accuracy_finder(model_obj=model_obj,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
            logging.info(f"Best Model Score : {model_score} New Trained Model Score :{trained_model_score}")
            
            if model_score<trained_model_score:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)

            return model_evaluation_artifact
        except Exception as e:
            raise InsuranceException(e,sys) from e