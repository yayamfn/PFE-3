import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODELS_DIR = 'models'

model = None
scaler = None
selector = None
feature_names = []

try:
    model_path = os.path.join(MODELS_DIR, 'model_xgb.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    selector_path = os.path.join(MODELS_DIR, 'selector.pkl')
    features_path = os.path.join(MODELS_DIR, 'feature_names.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(selector_path):
        raise FileNotFoundError(f"Selector file not found: {selector_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature names file not found: {features_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    with open(features_path) as f:
        feature_names = json.load(f)
    logger.info("Modèle, scaler, selector et feature_names chargés avec succès.")
    logger.info("Features attendues pour l'API: %s", feature_names)
except FileNotFoundError as e:
    logger.error(f"Erreur critique: {e}. Veuillez exécuter train.py d'abord pour générer les actifs nécessaires.")
except Exception as e:
    logger.error(f"Erreur inattendue lors du chargement des actifs: {e}")

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de supervision pour vérifier que l'API et les modèles sont chargés."""
    status = {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "selector_loaded": selector is not None,
        "feature_names_loaded": bool(feature_names)
    }
    return jsonify(status), 200 if all(status.values()) else 503

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or selector is None or not feature_names:
        logger.error("Tentative de prédiction alors que les actifs ne sont pas chargés.")
        return jsonify({"error": "Les modèles ou les actifs de prétraitement ne sont pas disponibles. Veuillez contacter l'administrateur ou exécuter le script d'entraînement."}), 503

    try:
        data = request.get_json(force=True)

        if not isinstance(data, dict):
             return jsonify({"error": "Les données d'entrée doivent être un objet JSON unique (dictionnaire)."}), 400

        input_values = [data.get(feat, 0.0) for feat in feature_names]
        
        for i, val in enumerate(input_values):
            if not isinstance(val, (int, float)):
                try:
                    input_values[i] = float(val)
                except ValueError:
                    logger.error(f"Valeur non numérique pour la feature '{feature_names[i]}': {val}")
                    return jsonify({"error": f"Valeur non numérique '{val}' pour la feature '{feature_names[i]}'. Les features doivent être des nombres."}), 400

        df = pd.DataFrame([input_values], columns=feature_names)

        X_scaled = scaler.transform(df)
        X_selected = selector.transform(X_scaled)

        pred_encoded = model.predict(X_selected)[0]
        prediction_result = int(pred_encoded)

        response = {'prediction': prediction_result}
        if hasattr(model, 'predict_proba'):
            proba_all_classes = model.predict_proba(X_selected)[0]
            probability_of_predicted_class = proba_all_classes[prediction_result]
            probability_of_parkinsons = proba_all_classes[1] if len(proba_all_classes) > 1 else probability_of_predicted_class
            
            response['probability_predicted_class'] = float(probability_of_predicted_class)
            response['probability_parkinsons'] = float(probability_of_parkinsons)
        
        return jsonify(response)

    except Exception as e:
        logger.error("Erreur interne lors de la prédiction: %s", e, exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur. Veuillez contacter l\'administrateur.'}), 500

if __name__ == '__main__':
    if model is None or scaler is None or selector is None or not feature_names:
        logger.warning("L'API démarre, mais les actifs du modèle ne sont pas chargés. Les prédictions échoueront.")
    logger.info("Démarrage de l'application Flask sur http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
