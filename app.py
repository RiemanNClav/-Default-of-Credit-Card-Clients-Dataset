from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Cargar el conjunto de datos (ajusta el nombre del archivo o la ruta según tu caso)
df = pd.read_csv('artifacts/train.csv')

# Obtener valores únicos de las categorías
companies = df['company'].unique()
toppings = df['topping'].unique()
variants = df['variant'].unique()
sizes = df['size'].unique()
extra_sauces = df['extra_sauce'].unique()
extra_cheeses = df['extra_cheese'].unique()
extra_mushrooms = df['extra_mushrooms'].unique()

# Ruta para la página de inicio
@app.route('/')
def index():
    return render_template('index.html') 

# Ruta para predecir datos
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html',
                               companies=companies,
                               toppings=toppings,
                               variants=variants,
                               sizes=sizes,
                               extra_sauces=extra_sauces,
                               extra_cheeses=extra_cheeses,
                               extra_mushrooms=extra_mushrooms)
    else:
        data = CustomData(
            diameter=float(request.form.get('diameter')),
            company=request.form.get('company'),
            topping=request.form.get('topping'),
            variant=request.form.get('variant'),
            size=request.form.get('size'),
            extra_sauce=request.form.get('extra_sauce'),
            extra_cheese=request.form.get('extra_cheese'),
            extra_mushrooms=request.form.get('extra_mushrooms')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', 
                               results=results[0],
                               companies=companies,
                               toppings=toppings,
                               variants=variants,
                               sizes=sizes,
                               extra_sauces=extra_sauces,
                               extra_cheeses=extra_cheeses,
                               extra_mushrooms=extra_mushrooms)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
