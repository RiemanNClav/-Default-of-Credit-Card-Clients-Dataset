import sys
from dataclasses import dataclass
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

class DataCleaning(object):
    def __init__(self):
        pass

    def initiate_data_cleaning(self, train_path, test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        #train_df = train_path
        #test_df = test_path

        train_df["SEX"] = train_df["SEX"].astype("str")
        train_df["EDUCATION"] = train_df["EDUCATION"].astype("str")
        train_df["MARRIAGE"] = train_df["MARRIAGE"].astype("str")
        train_df["AGE"] = train_df["AGE"].astype("str")
        train_df['EDUCATION'] = train_df['EDUCATION'].replace({'0': '4', '5': '4', '6': '4'})
        train_df.rename(columns={'default payment next month': 'DEFAULT','PAY_0': 'PAY_1'}, inplace=True)


        test_df["SEX"] = test_df["SEX"].astype("str")
        test_df["EDUCATION"] = test_df["EDUCATION"].astype("str")
        test_df["MARRIAGE"] = test_df["MARRIAGE"].astype("str")
        test_df["AGE"] = test_df["AGE"].astype("str")
        test_df['EDUCATION'] = test_df['EDUCATION'].replace({'0': '4', '5': '4', '6': '4'})
        test_df.rename(columns={'default payment next month': 'DEFAULT','PAY_0': 'PAY_1'}, inplace=True)
        
        return train_df, test_df

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            
            numerical_columns =['LIMIT_BAL','PAY_1','PAY_2', 'PAY_3', 'PAY_4', "PAY_5", "PAY_6"]
                                # 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                                # 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                                # 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']

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

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                    
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df= train_path
            test_df= test_path

            logging.info(f"Read train and test data completed")

            logging.info(f"Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="DEFAULT"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)[['LIMIT_BAL','PAY_1','PAY_2', 'PAY_3', 'PAY_4','PAY_5', 'PAY_6', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']]

            print(input_feature_train_df)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)[['LIMIT_BAL','PAY_1','PAY_2', 'PAY_3', 'PAY_4','PAY_5', 'PAY_6', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']]
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df).toarray()

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)