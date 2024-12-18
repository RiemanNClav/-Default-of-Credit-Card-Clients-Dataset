import os
import sys
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
#metricas, parametros
from sklearn.model_selection import  GridSearchCV

#mlflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    trained_model_file_path=os.path.join(base_dir, "artifacts", "model.pkl")
    preprocessing_file_path=os.path.join(base_dir, "artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def models_params(self):
        models_params_ = {
            "Random Forest": {
                "model": RandomForestClassifier(),
                "params": {
                    'n_estimators': [50],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(),
                "params": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                }
            },
            "Logistic Regression": {
                "model": LogisticRegression(),
                "params": {
                    'penalty': ['l1', 'l2'],
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga']
                }
            },
            "Support Vector Machine": {
                "model": SVC(),
                "params": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                }
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier(),
                "params": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                }
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(),
                "params": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                }
            },
            "AdaBoost": {
                "model": AdaBoostClassifier(),
                "params": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.5, 1.0],
                }
            },
            "XGBoost": {
                "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "params": {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1],
                }
            }
        }
        return models_params_
            

    def initiate_model_trainer(self, train_array, test_array, run_id_):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            report_r2 = {}
            report_params = {}
            report_metrics = {}
            mlflow_models_info = []


            for model_name, config in self.models_params().items():
                model = config["model"]
                params = config["params"]

                gs = GridSearchCV(model, params)
                gs.fit(X_train, y_train)

                # Ajustar modelo con los mejores parámetros
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_test_pred)
                precision_multi = precision_score(y_test, y_test_pred, average='weighted')
                recall_multi = recall_score(y_test, y_test_pred, average='weighted')
                f1_multi = f1_score(y_test, y_test_pred, average='weighted')

                print(f"model_name = {accuracy}")
                print('\n')

                # guardar métricas
                report_r2[model_name] = accuracy
                report_params[model_name] = gs.best_params_
                report_metrics[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision_multi,
                    "recall": recall_multi,
                    "F1": f1_multi,
                }

                # Registrar modelo en MLflow
                with mlflow.start_run(run_id=run_id_):
                    mlflow.sklearn.log_model(model, f"{model_name}/model")
                    accuracy = report_metrics[model_name]["accuracy"]
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision_multi)
                    mlflow.log_metric("recall", recall_multi)
                    mlflow.log_metric("F1", f1_multi)
                    # mlflow.log_params(gs.best_params_)

                    version = mlflow.register_model(f"runs:/{run_id_}/{model_name}/model", f"{model_name}__model")
                    mlflow_models_info.append((model_name, version, accuracy))

            # Lógica para determinar el modelo campeón
            best_model_name, best_model_version, best_accuracy = max(
                mlflow_models_info, key=lambda x: x[2]
            )

            if best_accuracy < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Champion Model: {best_model_name} (Version: {best_model_version})")

            client = MlflowClient()

            client.set_registered_model_alias(best_model_version.name, "champion", version.version,)

            client.set_registered_model_tag(best_model_version.name, "prediction_credits", "true",)

            # guardar el modelo campeón localmente
            best_model_obj = self.models_params()[best_model_name]["model"]
            best_model_obj.set_params(**report_params[best_model_name])
            best_model_obj.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_obj,
            )

        except Exception as e:
            raise CustomException(e, sys)
