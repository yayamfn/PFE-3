import os
import warnings
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import optuna
import shap
import gradio as gr
from flask import Flask, request, jsonify
import joblib
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

def load_data(url):
    """
    Load the Parkinson's dataset from the given URL.
    
    Parameters:
    url (str): URL to the dataset.
    
    Returns:
    X (pd.DataFrame): Features.
    y (np.array): Target variable.
    le (LabelEncoder): Label encoder for the target.
    """
    data = pd.read_csv(url)
    X = data.drop(columns=['name', 'status'])
    y = data['status']
    le = LabelEncoder()
    y = le.fit_transform(y).astype(int)
    logger.info("Valeurs uniques dans y: %s", np.unique(y))
    return X, y, le

def plot_class_distribution(y):
    """
    Plot the distribution of classes and save the figure.
    
    Parameters:
    y (np.array): Target variable.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution des Classes')
    plt.savefig(os.path.join('figures', 'class_distribution.png'))
    plt.close()

def plot_correlation_matrix(X):
    """
    Plot the correlation matrix of features and save the figure.
    
    Parameters:
    X (pd.DataFrame): Features.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), cmap='coolwarm', annot=False)
    plt.title('Matrice de Corrélation')
    plt.savefig(os.path.join('figures', 'correlation_matrix.png'))
    plt.close()

def preprocess_data(X, y):
    """
    Preprocess the data: split, scale, apply SMOTE, and select features.
    
    Parameters:
    X (pd.DataFrame): Features.
    y (np.array): Target variable.
    
    Returns:
    X_train_selected (np.array): Selected training features.
    X_test_selected (np.array): Selected testing features.
    y_train_smote (np.array): Balanced training target.
    y_test (np.array): Testing target.
    scaler (StandardScaler): Scaler object.
    selector (SelectKBest): Feature selector object.
    X_test (pd.DataFrame): Original testing features.
    """
    warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    logger.info("Classes après SMOTE: %s", np.unique(y_train_smote, return_counts=True))
    selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
    X_test_selected = selector.transform(X_test_scaled)
    return X_train_selected, X_test_selected, y_train_smote, y_test, scaler, selector, X_test

def objective_rf(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                   class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold = X_train[train_idx]
        y_fold = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        if len(np.unique(y_fold)) < 2:
            logger.warning("Saut d'un pli avec une seule classe")
            continue
        model.fit(X_fold, y_fold)
        score = model.score(X_val, y_val)
        scores.append(score)
    return np.mean(scores) if scores else np.nan

def objective_xgb(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                          learning_rate=learning_rate, random_state=42, 
                          objective='binary:logistic')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold = X_train[train_idx]
        y_fold = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        if len(np.unique(y_fold)) < 2:
            logger.warning("Saut d'un pli avec une seule classe")
            continue
        model.fit(X_fold, y_fold)
        score = model.score(X_val, y_val)
        scores.append(score)
    return np.mean(scores) if scores else np.nan

def train_models(X_train, y_train):
    """
    Train Random Forest and XGBoost models with hyperparameter optimization.
    
    Parameters:
    X_train (np.array): Training features.
    y_train (np.array): Training target.
    
    Returns:
    model_rf_optimized (RandomForestClassifier): Optimized RF model.
    model_xgb_optimized (XGBClassifier): Optimized XGBoost model.
    """
    # Random Forest optimization
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=50)
    model_rf_optimized = RandomForestClassifier(**study_rf.best_params, class_weight='balanced', random_state=42)
    model_rf_optimized.fit(X_train, y_train)
    
    # XGBoost optimization
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=50)
    model_xgb_optimized = XGBClassifier(**study_xgb.best_params, random_state=42, objective='binary:logistic')
    model_xgb_optimized.fit(X_train, y_train)
    
    return model_rf_optimized, model_xgb_optimized

def evaluate_model(model, X_test, y_test, model_name, le):
    """
    Evaluate the model and save confusion matrix and ROC curve plots.
    
    Parameters:
    model: Trained model.
    X_test (np.array): Testing features.
    y_test (np.array): Testing target.
    model_name (str): Name of the model for file naming.
    le (LabelEncoder): Label encoder for class names.
    """
    y_pred = model.predict(X_test)
    logger.info("%s:\n%s", model_name, classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Précision du modèle %s : %.2f%%", model_name, accuracy * 100)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.savefig(os.path.join('figures', f'confusion_matrix_{model_name}.png'))
    plt.close()
    
    # ROC Curve
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title(f'Courbe ROC - {model_name}')
        plt.legend()
        plt.savefig(os.path.join('figures', f'roc_curve_{model_name}.png'))
        plt.close()

def interpret_model(model, X_test, feature_names):
    """
    Interpret the model using SHAP and save the summary plot.
    
    Parameters:
    model: Trained model.
    X_test (np.array): Testing features.
    feature_names (list): List of feature names.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join('figures', 'shap_summary.png'))
    plt.close()

def run_internal_tests(model, X_test, y_test, scaler, selector, feature_names):
    """
    Run internal tests on sample data.
    
    Parameters:
    model: Trained model.
    X_test (pd.DataFrame): Original testing features.
    y_test (np.array): Testing target.
    scaler (StandardScaler): Scaler object.
    selector (SelectKBest): Feature selector object.
    feature_names (list): List of feature names.
    """
    logger.info("Exécution des tests internes...")
    for i in range(5):
        sample = X_test.iloc[i].to_dict()
        true_label = y_test[i]
        input_data = np.array([sample[feature] for feature in feature_names]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)
        prediction = model.predict(input_selected)[0]
        logger.info("Échantillon %d: Prédit %d, Vrai %d", i, prediction, true_label)
    logger.info("Tests internes terminés.")

def save_models(model, scaler, selector, feature_names):
    """
    Save the model, scaler, selector, and feature names to disk.
    
    Parameters:
    model: Trained model.
    scaler (StandardScaler): Scaler object.
    selector (SelectKBest): Feature selector object.
    feature_names (list): List of feature names.
    """
    joblib.dump(model, 'model_xgb.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(selector, 'selector.pkl')
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

if __name__ == '__main__':
    # Load data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
    X, y, le = load_data(url)
    
    # Plot distributions and correlations
    plot_class_distribution(y)
    plot_correlation_matrix(X)
    
    # Preprocess data
    X_train_selected, X_test_selected, y_train_smote, y_test, scaler, selector, X_test = preprocess_data(X, y)
    
    # Train models
    model_rf_optimized, model_xgb_optimized = train_models(X_train_selected, y_train_smote)
    
    # Evaluate models
    evaluate_model(model_rf_optimized, X_test_selected, y_test, 'RandomForest', le)
    evaluate_model(model_xgb_optimized, X_test_selected, y_test, 'XGBoost', le)
    
    # Interpret model with SHAP
    selected_feature_names = X.columns[selector.get_support()]
    interpret_model(model_xgb_optimized, X_test_selected, selected_feature_names)
    
    # Run internal tests
    run_internal_tests(model_xgb_optimized, X_test, y_test, scaler, selector, list(X.columns))
    
    # Save models
    save_models(model_xgb_optimized, scaler, selector, list(X.columns))
    
    app = Flask(__name__)
logger = logging.getLogger(__name__)

# Chargement des objets
model      = joblib.load('model_xgb.pkl')
scaler     = joblib.load('scaler.pkl')
selector   = joblib.load('selector.pkl')
with open('feature_names.json') as f:
    feature_names = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or not all(feat in data for feat in feature_names):
            return jsonify({'error': 'Veuillez fournir toutes les features requises.'}), 400
        # Construction du DataFrame
        # 1) Construction d'un DataFrame avec colonnes nommées
        df = pd.DataFrame([{feat: data.get(feat, 0) 
                            for feat in feature_names}])
        # 2) Mise à l’échelle et sélection
        X_scaled   = scaler.transform(df)
        X_selected = selector.transform(X_scaled)
        # 3) Prédiction
        pred = int(model.predict(X_selected)[0])
        return jsonify({'prediction': pred})
    except Exception as e:
        logger.error("Erreur dans la prédiction: %s", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)