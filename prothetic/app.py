from flask import Flask, request, render_template, redirect, url_for, session as flask_session, send_file
import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import os

app = Flask(__name__)
app.secret_key = 'prothetic_secret_key'
metadata = bootstrap_project(Path.cwd())

project_path = Path("prothetic")


@app.route('/', methods=['GET'])
def home():
    """
    Route pour la page d'accueil.
    """
   # Récupère les messages flash
    error = flask_session.pop('error', None)
    success = flask_session.pop('success', None)

    # Passe uniquement les messages non nulles
    messages = {}
    if error:
        messages['error'] = error
    if success:
        messages['success'] = success

    return render_template('index.html', messages=messages)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Route pour effectuer des prédictions.
    """
    if request.method == 'POST':
        try:
            # Récupère le fichier téléchargé et le lit dans un DataFrame pandas
            file = request.files['file']
            raw_data_df = pd.read_csv(file)

            # Sauvegarde les données brutes dans un fichier CSV
            raw_data_df.to_csv(
                "data/01_raw/raw_data_to_predict.csv", index=False)

            # Crée une session Kedro et exécute le pipeline de prédiction
            with KedroSession.create(project_path=".") as session:
                session.run(pipeline_name="model_prediction")

            # Lit les données transformées et les prédictions à partir des fichiers CSV générés par Kedro
            transformed_data = pd.read_csv(
                "data/07_model_input/data_to_predict.csv")
            data_predicted = pd.read_csv(
                "data/07_model_output/predictions.csv")

            # Concatène les données transformées et les prédictions pour l'affichage
            data_to_predict = pd.concat(
                [transformed_data, data_predicted], axis=1)

            # Préparation des données pour Chart.js
            values = {
                '1000_Hz': {
                    'before': data_to_predict['before_exam_1000_Hz'].tolist(),
                    'after': data_to_predict['after_exam_1000_Hz'].tolist()
                },
                '2000_Hz': {
                    'before': data_to_predict['before_exam_2000_Hz'].tolist(),
                    'after': data_to_predict['after_exam_2000_Hz'].tolist()
                },
                '4000_Hz': {
                    'before': data_to_predict['before_exam_4000_Hz'].tolist(),
                    'after': data_to_predict['after_exam_4000_Hz'].tolist()
                }
            }

            # Récupère les noms de colonnes pour l'affichage
            labels = data_to_predict.index.tolist()

            # Rend la page de résultats avec les données concaténées
            return render_template('results.html', tables=[data_to_predict.to_html(classes='data')], labels=labels, values=values)
        except Exception as e:
            # En cas d'erreur, enregistre le message d'erreur dans la session et redirige
            flask_session['error'] = str(e)
            return redirect(url_for('home'))
    # Affiche la page pour télécharger un fichier pour la prédiction
    return render_template('results.html')


@app.route('/aggregate', methods=['GET', 'POST'])
def aggregate():
    """
    Route pour agréger des données et entraîner le modèle.
    """
    if request.method == 'POST':
        try:
            # Récupère le fichier téléchargé et le lit dans un DataFrame pandas
            file = request.files['file']
            aggregate_data_df = pd.read_csv(file)

            # Colonnes nécessaires pour l'entraînement du modèle
            columns = [
                'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz', 'before_exam_1000_Hz',
                'before_exam_2000_Hz', 'before_exam_4000_Hz', 'before_exam_8000_Hz', 'after_exam_125_Hz',
                'after_exam_250_Hz', 'after_exam_500_Hz', 'after_exam_1000_Hz', 'after_exam_2000_Hz',
                'after_exam_4000_Hz', 'after_exam_8000_Hz'
            ]

            # Vérifie que les données d'agrégation contiennent toutes les colonnes nécessaires
            missing_columns = [
                col for col in columns if col not in aggregate_data_df.columns]
            if missing_columns:
                raise ValueError(f"Les colonnes suivantes sont manquantes dans les données d'agrégation : {', '.join(missing_columns)}")

            tonal_exams_path = "data/01_raw/tonal_exams.csv"

            # Vérifie si le fichier tonal_exams.csv existe déjà et le lit, sinon crée un DataFrame vide
            if os.path.exists(tonal_exams_path):
                existing_data_df = pd.read_csv(tonal_exams_path)
            else:
                existing_data_df = pd.DataFrame()

            # Concatène les nouvelles données avec les données existantes
            combined_df = pd.concat(
                [existing_data_df, aggregate_data_df], ignore_index=True)

            # Sauvegarde les données combinées dans le fichier tonal_exams.csv
            combined_df.to_csv(tonal_exams_path, index=False)

            # Crée une session Kedro et exécute le pipeline par défaut pour entraîner le modèle
            with KedroSession.create(project_path=".") as session:
                session.run(pipeline_name="__default__")

            # Enregistre le message de succès dans la session et redirige
            flask_session['success'] = "Model training completed successfully!"
            return redirect(url_for('home'))
        except Exception as e:
            # Enregistre le message d'erreur dans la session et redirige
            flask_session['error'] = str(e)
            return redirect(url_for('home'))
    # Affiche la page pour télécharger un fichier pour l'agrégation
    return render_template('aggregate.html')


@app.route('/download', methods=['GET'])
def download():
    """
    Route pour télécharger les prédictions au format CSV.
    """
    try:
        # Chemin du fichier CSV contenant les prédictions
        transformed_data = pd.read_csv(
            "data/07_model_input/data_to_predict.csv")
        predictions_csv = "data/07_model_output/predictions.csv"

        # Vérifie si le fichier existe
        if not os.path.exists(predictions_csv):
            flask_session['error'] = "Le fichier des prédictions n'existe pas."
            return redirect(url_for('home'))

        data_predicted = pd.concat(
            [transformed_data, pd.read_csv(predictions_csv)], axis=1)

        # Sauvegarde les données concaténées dans un fichier temporaire
        temp_file = "data/07_model_output/combined_predictions.csv"
        data_predicted.to_csv(temp_file, index=False)

        # Envoie le fichier CSV en pièce jointe
        return send_file(temp_file, as_attachment=True)
    except Exception as e:
        flask_session['error'] = str(e)
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
