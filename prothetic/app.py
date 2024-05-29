from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import os

app = Flask(__name__)
metadata = bootstrap_project(Path.cwd())

project_path = Path("prothetic")


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        raw_data_df = pd.read_csv(file)

        raw_data_df.to_csv("data/01_raw/raw_data_to_predict.csv", index=False)

        with KedroSession.create(project_path=".") as session:
            session.run(pipeline_name="model_prediction")


        transformed_data = pd.read_csv("data/07_model_input/data_to_predict.csv")
        data_predicted = pd.read_csv("data/07_model_output/predictions.csv")

        data_to_predict = pd.concat([transformed_data, data_predicted], axis=1)
        

        return render_template('results.html', tables=[data_to_predict.to_html(classes='data')], titles=data_to_predict.columns.values)
    except Exception as e:
        return redirect(url_for('home', error=str(e)))


@app.route('/aggregate', methods=['GET', 'POST'])
def aggregate():
    if request.method == 'POST':
        try:
            file = request.files['file']
            aggregate_data_df = pd.read_csv(file)

            tonal_exams_path = "data/01_raw/tonal_exams.csv"

            if os.path.exists(tonal_exams_path):
                existing_data_df = pd.read_csv(tonal_exams_path)
            else:
                existing_data_df = pd.DataFrame()

            combined_df = pd.concat(
                [existing_data_df, aggregate_data_df], ignore_index=True)

            combined_df.to_csv(tonal_exams_path, index=False)

            with KedroSession.create(project_path=".") as session:
                session.run(pipeline_name="__default__")

            return redirect(url_for('home', success="Model training completed successfully!"))
        except Exception as e:
            return redirect(url_for('home', error=str(e)))
    return render_template('aggregate.html')


if __name__ == "__main__":
    app.run(port=8000, debug=True)
