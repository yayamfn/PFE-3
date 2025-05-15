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

# 1. Chargement et Analyse Exploratoire
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')
X = data.drop(columns=['name', 'status'])
y = data['status']

# Encodage explicite des labels
le = LabelEncoder()
y = le.fit_transform(y).astype(int)
print("Valeurs uniques dans y:", np.unique(y))

# Visualisations
plt.figure(figsize=(8, 6))
sns.countplot(x='status', data=data)
plt.title('Distribution des Classes')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap='coolwarm', annot=False)
plt.title('Matrice de Corrélation')
plt.show()

# 2. Prétraitement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Équilibrage avec SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print("Classes après SMOTE:", np.unique(y_train_smote, return_counts=True))

# 3. Sélection de Caractéristiques
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
X_test_selected = selector.transform(X_test_scaled)

# 4. Entraînement Initial
model_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
model_rf.fit(X_train_selected, y_train_smote)

model_xgb = XGBClassifier(random_state=42, objective='binary:logistic')
model_xgb.fit(X_train_selected, y_train_smote)

# 5. Optimisation des Hyperparamètres
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                   class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train_selected, y_train_smote):
        X_fold = X_train_selected[train_idx]
        y_fold = y_train_smote[train_idx]
        X_val = X_train_selected[val_idx]
        y_val = y_train_smote[val_idx]
        print("Classes dans y_fold:", np.unique(y_fold))
        if len(np.unique(y_fold)) < 2:
            print("Saut d'un pli avec une seule classe")
            continue
        model.fit(X_fold, y_fold)
        score = model.score(X_val, y_val)
        scores.append(score)
    return np.mean(scores) if scores else np.nan

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=50)
model_rf_optimized = RandomForestClassifier(**study_rf.best_params, class_weight='balanced', random_state=42)
model_rf_optimized.fit(X_train_selected, y_train_smote)

def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                          learning_rate=learning_rate, random_state=42, 
                          objective='binary:logistic')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train_selected, y_train_smote):
        X_fold = X_train_selected[train_idx]
        y_fold = y_train_smote[train_idx]
        X_val = X_train_selected[val_idx]
        y_val = y_train_smote[val_idx]
        print("Classes dans y_fold:", np.unique(y_fold))
        if len(np.unique(y_fold)) < 2:
            print("Saut d'un pli avec une seule classe")
            continue
        model.fit(X_fold, y_fold)
        score = model.score(X_val, y_val)
        scores.append(score)
    return np.mean(scores) if scores else np.nan

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)
model_xgb_optimized = XGBClassifier(**study_xgb.best_params, random_state=42, 
                                    objective='binary:logistic')
model_xgb_optimized.fit(X_train_selected, y_train_smote)

# 6. Évaluation
y_pred_rf = model_rf_optimized.predict(X_test_selected)
y_pred_xgb = model_xgb_optimized.predict(X_test_selected)

print("Random Forest Optimisé:\n", classification_report(y_test, y_pred_rf))
print("XGBoost Optimisé:\n", classification_report(y_test, y_pred_xgb))
accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"Précision du modèle en cours d'utilisation : {accuracy * 100:.2f}%")

# Matrice de Confusion
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion - XGBoost')
plt.show()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, model_xgb_optimized.predict_proba(X_test_selected)[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - XGBoost')
plt.legend()
plt.show()

# 7. Interprétation avec SHAP (optionnel, pour démonstration)
explainer = shap.TreeExplainer(model_xgb_optimized)
shap_values = explainer.shap_values(X_test_selected)
shap.summary_plot(shap_values, X_test_selected, feature_names=X.columns[selector.get_support()])

# 8. Tests Internes
print("Exécution des tests internes...")
for i in range(5):
    sample = X_test.iloc[i].to_dict()
    true_label = y_test[i]
    input_data = np.array([sample[feature] for feature in X.columns]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    input_selected = selector.transform(input_scaled)
    prediction = model_xgb_optimized.predict(input_selected)[0]
    print(f"Échantillon {i}: Prédit {prediction}, Vrai {true_label}")
print("Tests internes terminés.")

# 9. Déploiement
# Sauvegarde des modèles et objets de prétraitement
joblib.dump(model_xgb_optimized, 'model_xgb.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')

# API Flask
app = Flask(__name__)

feature_names = list(X.columns)
print("Noms des caractéristiques:", feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Vérifier que toutes les caractéristiques sont présentes
        input_data = [data.get(feature, 0) for feature in feature_names]
        input_data = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)
        prediction = model_xgb_optimized.predict(input_selected)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Lancement du serveur Flask...")
    app.run(debug=False, host='0.0.0.0', port=5000)

# Démo Gradio (optionnel)
def predict_parkinson(*features):
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    input_selected = selector.transform(input_scaled)
    prediction = model_xgb_optimized.predict(input_selected)[0]
    return "Parkinson" if prediction[0] == 1 else "Sain"

interface = gr.Interface(fn=predict_parkinson, inputs=[gr.Number(label=feature) for feature in feature_names], outputs="text")
interface.launch()