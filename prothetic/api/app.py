from flask import Blueprint, request, jsonify, render_template
import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import pickle
import os

bp = Blueprint('app', __name__)

project_path = Path(__file__).resolve().parents[1]  # Adjust path to parent directory where pyproject.toml is located
env = "local"  # or your desired environment

# Bootstrap the Kedro project
metadata = bootstrap_project(project_path)

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

@bp.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        raw_data_df = pd.read_csv(file)
        
        # Ensure the raw data directory exists
        raw_data_directory = project_path / "data" / "raw"
        ensure_directory_exists(raw_data_directory)
        
        # Save data to appropriate location for prediction
        raw_data_df.to_csv(raw_data_directory / "raw_data_to_predict.csv", index=False)
        
        # Run Kedro pipeline for prediction
        with KedroSession.create(metadata.package_name, project_path, env=env) as session:
            session.run(pipeline_name="model_prediction")

        # Load the transformed and predicted data
        context = session.load_context()
        data_to_predict = context.catalog.load("data_to_predict")
        
        # Load the trained model and make predictions
        with open(project_path / "data" / "models" / "final_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        predictions = model.predict(data_to_predict)
        
        # Add predictions to the DataFrame
        data_to_predict['Predictions'] = predictions.tolist()

        return render_template('results.html', tables=[data_to_predict.to_html(classes='data')], titles=data_to_predict.columns.values)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/aggregate', methods=['GET', 'POST'])
def aggregate():
    if request.method == 'POST':
        try:
            file = request.files['file']
            aggregate_data_df = pd.read_csv(file)
            
            # Ensure the tonal exams data directory exists
            raw_data_directory = project_path / "data" / "raw"
            ensure_directory_exists(raw_data_directory)
            
            # Load existing tonal exams data
            tonal_exams_path = raw_data_directory / "tonal_exams.csv"
            if tonal_exams_path.exists():
                existing_data_df = pd.read_csv(tonal_exams_path)
            else:
                existing_data_df = pd.DataFrame()
            
            # Aggregate the new data with the existing data
            combined_df = pd.concat([existing_data_df, aggregate_data_df], ignore_index=True)
            
            # Save the aggregated data
            combined_df.to_csv(tonal_exams_path, index=False)

            # Run Kedro pipelines for data processing and model training
            with KedroSession.create(metadata.package_name, project_path, env=env) as session:
                session.run(pipeline_name="collect_transformation")
                session.run(pipeline_name="model_training")

            return render_template('results.html', tables=[combined_df.to_html(classes='data')], titles=combined_df.columns.values)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template('aggregate.html')
