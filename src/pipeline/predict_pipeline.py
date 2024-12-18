import sys
import os
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
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        diameter: int,
        company: str,
        topping: str,
        variant: str,
        size: str,
        extra_sauce: str,
        extra_cheese: str,
        extra_mushrooms: str):

        self.diameter = diameter

        self.company = company

        self.topping = topping

        self.variant = variant

        self.size = size

        self.extra_sauce = extra_sauce

        self.extra_cheese = extra_cheese

        self.extra_mushrooms = extra_mushrooms

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "diameter": [self.diameter],
                "company": [self.company],
                "topping": [self.topping],
                "variant": [self.variant],
                "size": [self.size],
                "extra_sauce": [self.extra_sauce],
                "extra_cheese": [self.extra_cheese],
                "extra_mushrooms": [self.extra_mushrooms],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":

    data = CustomData(40,
                  "A",
                  "vegetables",
                  "thai_veggie",
                  "small",
                  "no",
                  "no",
                  "no")

    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline=PredictPipeline()
    print("Mid Prediction")
    results=predict_pipeline.predict(pred_df)
    print(f"price prediction: {results}")