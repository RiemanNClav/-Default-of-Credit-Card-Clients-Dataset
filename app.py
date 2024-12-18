from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

df = pd.read_csv('artifacts/train.csv')

SEX = df['SEX'].unique()
EDUCATION = ["0", "1", "2", "3", "4"]
MARRIAGE = df['MARRIAGE'].unique()
AGE = df['AGE'].unique()


@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html',
                               SEXs=SEX,
                               EDUCATIONs=EDUCATION,
                               MARRIAGsE=MARRIAGE,
                               AGEs=AGE)
    else:
        data = CustomData(
            limit_bal=int(request.form.get('LIMIT_BAL')),
            sex=request.form.get('SEX'),
            education=request.form.get('EDUCATION'),
            marriage=request.form.get('MARRIAGE'),
            age=request.form.get('AGE'),
            pay_1=float(request.form.get('PAY_1')),
            pay_2=float(request.form.get('PAY_2')),
            pay_3=float(request.form.get('PAY_3')),
            pay_4=float(request.form.get('PAY_4')),
            pay_5=float(request.form.get('PAY_5')),
            pay_6=float(request.form.get('PAY_6'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html',
                               result=results[0],
                               SEXs=SEX,
                               EDUCATIONs=EDUCATION,
                               MARRIAGsE=MARRIAGE,
                               AGEs=AGE)

if __name__ == "__main__":
    app.run(host="0.0.0.0")