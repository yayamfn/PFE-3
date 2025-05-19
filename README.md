# 🩺 Prédiction de la Maladie de Parkinson à l'aide du Machine Learning

Ce projet propose un pipeline complet pour la détection précoce de la maladie de Parkinson à partir de paramètres vocaux, en utilisant des méthodes de machine learning classiques (Random Forest, XGBoost). Il inclut :  
- Un code d’entraînement et d’évaluation
- Une API Flask pour la prédiction
- Une interface utilisateur Streamlit
- Des outils d’interprétabilité (SHAP)
- Des figures et graphes pour l’analyse

---

## 🚀 Fonctionnalités principales

- **Prétraitement avancé** : scaling, SMOTE, sélection de variables
- **Optimisation automatique** : Optuna pour les hyperparamètres
- **Interprétabilité** : SHAP pour expliquer les prédictions
- **API REST** : prédiction via endpoint `/predict`
- **Interface utilisateur** : saisie manuelle ou import JSON, visualisation des résultats et explications

---

## 📦 Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/yayamfn/PFE-3.git
   cd PFE-3
2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
  # Sur Windows
    venv\Scripts\activate
  # Sur Linux/Mac
    source venv/bin/activate
3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   
---

## 🛠️ Utilisation

1. **Entraîner les modèles**
     Lance le script d’entraînement pour générer les modèles et assets nécessaires :
   ```bash
   python train.py

2. **Lancer l’API Flask**
   ```bash
   python api.py
   
  L’API sera disponible sur http://localhost:5000 .

3. **Lancer l’interface utilisateur (Streamlit)**
   ```bash
   streamlit run interface_streamlit.py
   
---

## 📁 Structure du projet
  PFE-3/
│
├── train.py                  # Script d'entraînement et d'évaluation
├── api.py                    # API Flask pour la prédiction
├── interface_streamlit.py    # Interface utilisateur Streamlit
├── requirements.txt          # Dépendances Python
├── config.json               # Configuration de l’interface
├── models/                   # Modèles et objets de prétraitement sauvegardés
│   ├── model_xgb.pkl
│   ├── model_rf.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   └── feature_names.json
├── figures/                  # Graphes générés (matrice de confusion, ROC, SHAP, etc.)
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── confusion_matrix_XGBoost.png
│   ├── roc_curve_XGBoost.png
│   └── shap_summary_XGBoost.png
└── ...
   
---

## 📊 Exemples de figures générées
  Distribution des classes
class_distribution.pngMatrice de corrélation
correlation_matrix.pngMatrice de confusion
confusion_matrix_XGBoost.pngCourbe ROC
roc_curve_XGBoost.pngRésumé SHAP
shap_summary_XGBoost.png
   
---

## 👨‍💻 Auteurs et crédits
  Auteur principal : El moufannane Yahya
  Encadrant : Dr.Ibrahim Ouahbi
  Licence : Usage académique uniquement
     
---

## 📄 Licence
  Ce projet est sous licence MIT.
       
---

## 💡 Remarques
  º Pour toute question ou bug, ouvrir une issue sur GitHub ou contacter elmoufannaneyahya006@gmail.com 
  º Ce projet a été réalisé dans le cadre d’un PFE Licence Mathématiques Appliquées pour l’Intelligence Machine.
