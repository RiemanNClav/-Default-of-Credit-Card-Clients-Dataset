import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
        self,
        limit_bal: int,
        sex: str,
        education: str,
        marriage: str,
        age: str,
        pay_1: int,
        pay_2: int,
        pay_3: int,
        pay_4: int,
        pay_5: int,
        pay_6: int
    ):
        self.limit_bal = limit_bal
        self.sex = sex
        self.education = education
        self.marriage = marriage
        self.age = age
        self.pay_1 = pay_1
        self.pay_2 = pay_2
        self.pay_3 = pay_3
        self.pay_4 = pay_4
        self.pay_5 = pay_5
        self.pay_6 = pay_6


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "LIMIT_BAL": [self.limit_bal],
                "SEX": [self.sex],
                "EDUCATION": [self.education],
                "MARRIAGE": [self.marriage],
                "AGE": [self.age],
                "PAY_1": [self.pay_1],
                "PAY_2": [self.pay_2],
                "PAY_3": [self.pay_3],
                "PAY_4": [self.pay_4],
                "PAY_5": [self.pay_5],
                "PAY_6": [self.pay_6]}

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    data = CustomData(
        limit_bal=120000,
        sex='2',
        education='2',
        marriage='1',
        age='24',
        pay_1=-1,
        pay_2=2,
        pay_3=0,
        pay_4=0,
        pay_5=0,
        pay_6=2
    )

    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print(f"Default Payment Prediction: {results}")