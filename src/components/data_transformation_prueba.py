import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


class Metrics(BaseEstimator, TransformerMixin):
    
    def __init__(self, historial_pago, facturas, pagos, limite):
        self.metric_dict = {}
        self.historial_pago =   historial_pago
        self.facturas = facturas
        self.pagos = pagos
        self.limite = limite[0]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        X_transformed['promedio_retraso'] = X_transformed[self.historial_pago].mean(axis=1)
        X_transformed['maximo_retraso'] = X_transformed[self.historial_pago].max(axis=1)
        X_transformed['minimo_retraso'] = X_transformed[self.historial_pago].min(axis=1)
        X_transformed['cantidad_meses_con_retraso'] = (X_transformed[self.historial_pago] > 0).sum(axis=1)

        new_columns = []
        for i in range(len(self.pagos)):
            x = f'relacion_pago_factura_{i+1}'
            y = f'razon_deuda_{i+1}'
            X_transformed[x] = X_transformed[self.pagos[i]] / X_transformed[self.facturas[i]]
            X_transformed[y] = X_transformed[self.facturas[i]] / X_transformed[self.limite]
            new_columns.append(x)
            new_columns.append(y)

        X_transformed.fillna(0, inplace=True)

        X_transformed["target"] = X_transformed["default payment next month"]

        squema = ["LIMIT_BAL", 'SEX', 'EDUCATION', 'MARRIAGE', 
                       'AGE', "promedio_retraso", "maximo_retraso", "minimo_retraso", 
                       "cantidad_meses_con_retraso", "target"] + new_columns
        
        return X_transformed[squema]

@dataclass
class DataTransformationConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocessor_obj_file_path: str = os.path.join(base_dir, 'artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self, new_columns):
        '''
        This function is responsible for data transformation.
        '''
        try:
            
            numerical_columns =['ID', 'LIMIT_BAL','PAY_1',
                                'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
            
            historial_pago_ = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            facturas_ = ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            pagos_ = ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            limite_ = ["LIMIT_BAL"]
            

            num_pipeline = Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
   
            cat_pipeline_metrics = Pipeline(
                steps=[
                    ("metrics", Metrics(historial_pago=historial_pago_, facturas=facturas_, pagos=pagos_, limite=limite_)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")



            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                    
                ]
            )

            return cat_pipeline_metrics, preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df= train_path
            test_df= test_path

            preprocessing_obj_metrics, preprocessing_obj_X =self.get_data_transformer_object(new_columns)

            logging.info(f"Read train and test data completed")

            #TRANSFORMAMOS y

            target_column_name="target"


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)


             # TRANSFORMAMOS X -> metrics
            input_feature = preprocessing_obj_metrics.fit(input_feature_train_df)
            input_feature_train_arr=input_feature.transform(input_feature_train_df)
            input_feature_test_arr=input_feature.transform(input_feature_test_df)

            # TRANSFORMAMOS X -> preprocesamiento

            input_feature_train_arr=preprocessing_obj_X.fit_transform(input_feature_train_arr).toarray()
            input_feature_test_arr=preprocessing_obj_X.transform(input_feature_test_arr).toarray()


            train_arr = input_feature_train_arr.copy()
            test_arr = input_feature_test_arr.copy()
            train_arr['categoria'] = y_train
            test_arr['categoria'] = y_test

            categoria['label'] = y_train 
            categoria = categoria.dropna()
            categoria = categoria.drop_duplicates(['label'])
            categoria.to_csv(self.data_transformation_config.categorias_file_path, index=False)


            train_arr = np.array(train_arr)
            test_arr = np.array(test_arr)



            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj_X

            )



            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    
    cleaning = DataCleaning()
    train_data  = cleaning.initiate_data_cleaning(train_data)
    test_data  = cleaning.initiate_data_cleaning(test_data)