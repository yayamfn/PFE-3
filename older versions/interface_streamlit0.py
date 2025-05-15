import streamlit as st
import joblib
import pandas as pd
import shap
import json
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction de Parkinson", layout="wide")

@st.cache_resource
def load_objects():
    try:
        model = joblib.load('model_xgb.pkl')
        scaler = joblib.load('scaler.pkl')
        selector = joblib.load('selector.pkl')
        with open('feature_names.json') as f:
            feature_names = json.load(f)
        return model, scaler, selector, feature_names
    except FileNotFoundError as e:
        st.error(f"Erreur : Fichier non trouvé. Vérifiez que tous les fichiers sont présents : {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement des objets : {e}")
        return None, None, None, None

model, scaler, selector, feature_names = load_objects()

if model is None:
    st.stop()

st.title("Prédiction de la Maladie de Parkinson")
st.markdown("""
Cette application vous permet de prédire si une personne est atteinte de la maladie de Parkinson en fonction de caractéristiques vocales. Vous pouvez entrer les données manuellement ou importer un fichier JSON contenant les valeurs des caractéristiques.
""")

mode = st.selectbox("Choisissez le mode d'entrée", ["Saisie manuelle", "Importer un fichier JSON"])

def predict_and_explain(input_df):
    # Mise à l’échelle et sélection des variables
    X_scaled = scaler.transform(input_df)
    X_selected = selector.transform(X_scaled)
    # Prédiction
    proba = model.predict_proba(X_selected)[0, 1]
    pred = int(model.predict(X_selected)[0])
    # Explication SHAP
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(X_selected)
    base_value = explainer.expected_value
    # Récupérer les features sélectionnées
    if hasattr(selector, 'get_support'):
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    else:
        selected_features = feature_names
    # Si shap_val est une liste (cas binaire), prendre la classe positive
    if isinstance(shap_val, list) and len(shap_val) == 2:
        shap_val = shap_val[1]
        if isinstance(base_value, list):
            base_value = base_value[1]
    return pred, proba, base_value, shap_val[0], selected_features

if mode == "Saisie manuelle":
    st.header("Saisie Manuelle des Variables")
    st.markdown("Entrez les valeurs pour chaque caractéristique vocale. Assurez-vous que les valeurs sont dans des plages réalistes.")
    user_input = {}
    for feat in feature_names:
        user_input[feat] = st.number_input(feat, value=0.0, step=0.001, format="%.3f")
    if st.button("Prédire (Saisie Manuelle)"):
        if all(value == 0.0 for value in user_input.values()):
            st.warning("Veuillez remplir au moins une caractéristique avec une valeur non nulle.")
        else:
            with st.spinner("Calcul de la prédiction..."):
                input_df = pd.DataFrame([user_input])
                try:
                    pred, proba, base_value, shap_val, selected_features = predict_and_explain(input_df)
                    st.success(f"Prédiction : {'Malade' if pred==1 else 'Sain'} (probabilité : {proba:.2f})")
                    st.subheader("Explication Locale (SHAP)")
                    st.markdown("Ce graphique montre comment chaque caractéristique contribue à la prédiction de la maladie.")
                    selected_df = input_df[selected_features]
                    shap.force_plot(base_value, shap_val, selected_df.iloc[0], matplotlib=True, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig)
                    plt.clf()
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")
elif mode == "Importer un fichier JSON":
    st.header("Importer un Fichier JSON")
    st.markdown("Téléchargez un fichier JSON contenant les caractéristiques vocales. Les clés doivent correspondre aux noms des caractéristiques.")
    uploaded_file = st.file_uploader("Choisissez un fichier JSON", type="json")
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            missing = [feat for feat in feature_names if feat not in data]
            if missing:
                st.warning(f"Variables manquantes dans le JSON, complétées à 0 : {missing}")
            for feat in feature_names:
                if feat not in data:
                    data[feat] = 0.0
            input_df = pd.DataFrame([data])
            st.write("Données importées :", input_df)
            if st.button("Prédire (JSON)"):
                with st.spinner("Calcul de la prédiction..."):
                    try:
                        pred, proba, base_value, shap_val, selected_features = predict_and_explain(input_df)
                        st.success(f"Prédiction : {'Malade' if pred==1 else 'Sain'} (probabilité : {proba:.2f})")
                        st.subheader("Explication Locale (SHAP)")
                        st.markdown("Ce graphique montre comment chaque caractéristique contribue à la prédiction de la maladie.")
                        selected_df = input_df[selected_features]
                        shap.force_plot(base_value, shap_val, selected_df.iloc[0], matplotlib=True, show=False)
                        fig = plt.gcf()
                        st.pyplot(fig)
                        plt.clf()
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture ou de la prédiction du fichier JSON : {e}")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier JSON : {e}")

st.divider()
st.header("Importance Globale des Variables (SHAP)")
st.markdown("Ce graphique montre l'importance moyenne des caractéristiques pour la prédiction de la maladie.")
try:
    X_demo = pd.DataFrame(np.zeros((10, len(feature_names))), columns=feature_names)
    X_demo_scaled = scaler.transform(X_demo)
    X_demo_selected = selector.transform(X_demo_scaled)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_demo_selected)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_positive = shap_values[1]
    else:
        shap_values_positive = shap_values
    if hasattr(selector, 'get_support'):
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    else:
        selected_features = feature_names
    plt.clf()
    plt.figure()
    shap.summary_plot(shap_values_positive, X_demo_selected, feature_names=selected_features, show=False)
    st.pyplot(plt.gcf())
    plt.clf()
except Exception as e:
    st.warning(f"Impossible d'afficher l'importance globale : {e}")

st.divider()
st.info("""Pour tester avec un fichier JSON, créez un fichier contenant les variables attendues, par exemple :{ "MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, ... }""")

st.markdown("### À Propos")
st.markdown("Cette application utilise un modèle XGBoost pour prédire la maladie de Parkinson à partir de caractéristiques vocales. Les explications SHAP montrent l'impact de chaque caractéristique sur la prédiction.")