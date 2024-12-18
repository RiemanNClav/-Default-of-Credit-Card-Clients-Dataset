import os
import sys
from dataclasses import dataclass
import numpy as np

# modelos
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#metricas, parametros
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#mlflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
#from src.components.models import models_params

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def models_params(self):
        models_params_ = {
            "Random Forest": {
                "model": RandomForestRegressor(),
            "params": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                    }
                },
        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                    }
                },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(),
            "params": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                     }
                },
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}
                },
        "XGBRegressor": {
            "model": XGBRegressor(),
            "params": {
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators': [8,16,32,64,128,256]
                        }
                },
        "AdaBoost Regressor": {
            "model": AdaBoostRegressor(),
            "params": {
                'learning_rate':[.1,.01,0.5,.001],
                # 'loss':['linear','square','exponential'],
                'n_estimators': [8,16,32,64,128,256]
                        }
                },

        }
        return models_params_
        


    def initiate_model_trainer(self,train_array,test_array):
        try:
            experiment_name = "Modelo de Regresion- Roberto"
            artifact_repository = './mflow-run'

            mlflow.set_tracking_uri('http://127.0.0.1:5000/')
            #Inicializar cliente MLFLOW
            cliente = MlflowClient()
    
            try:
                experiment_id = cliente.create_experiment(experiment_name, artifact_location=artifact_repository)
            except:
                experiment_id = cliente.get_experiment_by_name(experiment_name).experiment_id

            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Entrena y registra los modelos
            report_r2 = {}
            report_params = {}

            for model_name, config in self.models_params().items():
                model = config["model"]
                params = config["params"]

                gs = GridSearchCV(model, params, cv=3)
            
                # Inicia una nueva ejecución en MLflow
                with mlflow.start_run(experiment_id=experiment_id, run_name=model_name) as run:

                        # Obtener identificación de ejecución
                    run_id = run.info.run_uuid
        
                     # Proporcione notas breves sobre el run.
                    MlflowClient().set_tag(run_id,
                                           "mlflow.note.content",
                                           "Este es un experimento para explorar diferentes modelos de aprendizaje automático para Campus Recruitment Dataset")
                    mlflow.sklearn.autolog()

                    # Definimos el custom tag
                    tags = {"Application": "Pizza price monitoring",
                            "release.candidate": "PMP",
                            "release.version": "2.2.0"}
                    
                    # Set Tag
                    mlflow.set_tags(tags)
                                    
                    # Log python environment details
                    #mlflow.log_artifact('PizzaPredictionV2/requirements.txt')
                    

                    gs.fit(X_train, y_train)
                    mlflow.log_params(gs.best_params_) #registro mejores parametros
                    mlflow.sklearn.log_model(gs.best_estimator_, "model") # registra el modelo

                    # Evalúa el modelo en el conjunto de test y registra la métrica
                    model.set_params(**gs.best_params_)
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)

                    # evalua metricas
                    mae = mean_absolute_error(y_test, y_test_pred)
                    mse = mean_squared_error(y_test, y_test_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    r2_square = r2_score(y_test, y_test_pred)

                    # registro de metricas
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("MSE", mse)
                    mlflow.log_metric("RMSE", rmse)
                    mlflow.log_metric("R2", r2_square)
                    mlflow.sklearn.log_model(model, model_name)

                    report_r2[model_name] = r2_square
                    report_params[model_name] = gs.best_params_

            # mejor r2
            best_model_score = max(sorted(report_r2.values()))

            #mejor modelo
            best_model_name = list(report_r2.keys())[
                list(report_r2.values()).index(best_model_score)
                ]
            #mejores parametros
            best_params = report_params[best_model_name]


            if best_model_score<0.6:

                raise CustomException("No best model found")
            
            else:

                logging.info(f"Best found model on both training and testing dataset")

            best_model_obj = self.models_params()[best_model_name]["model"]

            best_model_obj.set_params(**best_params)

            best_model_obj.fit(X_train, y_train)

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model_obj)
                


            predicted=best_model_obj.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return best_model_name, r2_square
            
            
        except Exception as e:
            raise CustomException(e,sys)
