# interface_streamlit.py

# Importation des bibliothèques nécessaires
import streamlit as st
import joblib
import pandas as pd
import shap
import json
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(page_title="Prédiction de Parkinson", layout="wide")

# Section 1: Chargement des objets (modèle, scaler, selector, noms des caractéristiques)
@st.cache_resource
def load_objects():
    """
    Charge les objets nécessaires pour le fonctionnement de l'application.
    - Modèle XGBoost
    - Scaler pour la normalisation
    - Sélecteur de caractéristiques
    - Noms des caractéristiques
    """
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

# Section 2: Fonction pour la prédiction et l'explication SHAP
def predict_and_explain(input_df, feature_names, scaler, selector, model):
    """
    Effectue la prédiction et génère les explications SHAP.
    - Prétraitement des données (scaling et sélection de caractéristiques)
    - Prédiction avec le modèle XGBoost
    - Calcul des valeurs SHAP pour l'explication locale
    """
    # Assurer que toutes les caractéristiques sont présentes
    all_features = set(feature_names)
    input_features = set(input_df.columns)
    missing = all_features - input_features
    if missing:
        for feat in missing:
            input_df[feat] = 0.0
    input_df = input_df[feature_names]  # Réordonner pour correspondre à l'ordre original
    
    # Prétraitement
    X_scaled = scaler.transform(input_df)
    X_selected = selector.transform(X_scaled)
    
    # Prédiction
    pred = int(model.predict(X_selected)[0])
    proba = float(model.predict_proba(X_selected)[0][1])
    
    # Obtenir les noms des caractéristiques sélectionnées
    if hasattr(selector, 'get_support'):
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    else:
        selected_features = feature_names
    
    # Explication locale SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_selected)
    
    # Pour la classification binaire
    base_value = explainer.expected_value
    shap_val = shap_values[0]  # Pour le premier (et seul) échantillon
    
    return pred, proba, base_value, shap_val, selected_features

# Section 3: Fonction pour l'interface de saisie manuelle
def display_manual_input(feature_names, scaler, selector, model):
    """
    Affiche l'interface pour la saisie manuelle des données.
    - Permet à l'utilisateur d'entrer les valeurs des caractéristiques vocales.
    - Gère la prédiction et l'affichage des résultats.
    """
    st.header("Saisie Manuelle des Variables")
    st.markdown("""
    Entrez les valeurs pour chaque caractéristique vocale. Ces nombres décrivent des aspects de la voix, comme sa fréquence ou son amplitude. Assurez-vous que les valeurs sont réalistes (pas toutes à zéro).
    """)
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
                    pred, proba, base_value, shap_val, selected_features = predict_and_explain(input_df, feature_names, scaler, selector, model)
                    st.success(f"Prédiction : {'Malade' if pred==1 else 'Sain'} (probabilité : {proba:.2f})")
                    st.markdown("""
                    **Résultat** : 
                    - **Malade** signifie qu'il y a une forte chance que la personne ait la maladie de Parkinson.
                    - **Sain** signifie qu'il y a peu de chances que la personne soit malade.
                    La probabilité (un nombre entre 0 et 1) montre à quel point le modèle est sûr de son résultat.
                    """)
                    st.subheader("Explication Locale (SHAP)")
                    st.markdown("""
                    Ce graphique montre pourquoi le modèle a donné ce résultat. Chaque barre représente une caractéristique de la voix :
                    - Les barres **rouges** augmentent la probabilité d'être malade.
                    - Les barres **bleues** diminuent cette probabilité, suggérant que la personne est saine.
                    Plus une barre est longue, plus cette caractéristique est importante pour la prédiction.
                    """)
                    selected_df = input_df[selected_features]
                    shap.force_plot(base_value, shap_val, selected_df.iloc[0], matplotlib=True, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig)
                    plt.clf()
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")

# Section 4: Fonction pour l'interface d'importation JSON
def display_json_import(feature_names, scaler, selector, model):
    """
    Affiche l'interface pour l'importation d'un fichier JSON.
    - Permet à l'utilisateur de télécharger un fichier JSON.
    - Gère la prédiction et l'affichage des résultats.
    """
    st.header("Importer un Fichier JSON")
    st.markdown("""
    Téléchargez un fichier JSON contenant les caractéristiques vocales. Le fichier doit inclure les noms des caractéristiques (comme "MDVP:Fo(Hz)") et leurs valeurs. Vous pouvez télécharger un exemple ci-dessous.
    """)
    if st.button("Télécharger un exemple de fichier JSON"):
        sample_json = {
            "MDVP:Fo(Hz)": 119.992,
            "MDVP:Fhi(Hz)": 157.302,
            "MDVP:Flo(Hz)": 74.997,
            # Ajoutez d'autres caractéristiques selon feature_names.json
        }
        st.download_button(
            label="Télécharger l'exemple",
            data=json.dumps(sample_json),
            file_name="sample.json",
            mime="application/json"
        )
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
                        pred, proba, base_value, shap_val, selected_features = predict_and_explain(input_df, feature_names, scaler, selector, model)
                        st.success(f"Prédiction : {'Malade' if pred==1 else 'Sain'} (probabilité : {proba:.2f})")
                        st.markdown("""
                        **Résultat** : 
                        - **Malade** signifie qu'il y a une forte chance que la personne ait la maladie de Parkinson.
                        - **Sain** signifie qu'il y a peu de chances que la personne soit malade.
                        La probabilité (un nombre entre 0 et 1) montre à quel point le modèle est sûr de son résultat.
                        """)
                        st.subheader("Explication Locale (SHAP)")
                        st.markdown("""
                        Ce graphique montre pourquoi le modèle a donné ce résultat. Chaque barre représente une caractéristique de la voix :
                        - Les barres **rouges** augmentent la probabilité d'être malade.
                        - Les barres **bleues** diminuent cette probabilité, suggérant que la personne est saine.
                        Plus une barre est longue, plus cette caractéristique est importante pour la prédiction.
                        """)
                        selected_df = input_df[selected_features]
                        shap.force_plot(base_value, shap_val, selected_df.iloc[0], matplotlib=True, show=False)
                        fig = plt.gcf()
                        st.pyplot(fig)
                        plt.clf()
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture ou de la prédiction du fichier JSON : {e}")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier JSON : {e}")

# Section 5: Fonction pour l'affichage de l'importance globale des variables
def display_global_importance(feature_names, scaler, selector, model):
    """
    Affiche l'importance globale des variables à l'aide d'un graphique SHAP.
    - Utilise des données factices pour calculer l'importance moyenne des caractéristiques.
    """
    st.divider()
    st.header("Importance Globale des Variables (SHAP)")
    st.markdown("""
    Ce graphique montre quelles caractéristiques vocales sont les plus importantes pour prédire la maladie de Parkinson. Les caractéristiques en haut ont un impact plus fort sur les prédictions.
    """)
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

# Section 6: Script principal
if __name__ == "__main__":
    # Chargement des objets
    model, scaler, selector, feature_names = load_objects()
    if model is None:
        st.stop()

    # Initialisation de l'interface utilisateur
    st.title("Prédiction de la Maladie de Parkinson")
    st.markdown("""
    Bienvenue ! Cette application vous aide à prédire si une personne a la maladie de Parkinson en analysant des caractéristiques vocales, comme la fréquence ou l'amplitude de la voix. Vous pouvez :
    - Entrer les données manuellement en remplissant des champs.
    - Importer un fichier JSON avec les valeurs.
    """)

    st.header("Comment utiliser cette application")
    st.markdown("""
    1. Choisissez si vous voulez entrer les données manuellement ou importer un fichier JSON dans le menu déroulant ci-dessous.
    2. Pour la saisie manuelle, remplissez les champs avec des nombres représentant les caractéristiques vocales.
    3. Pour importer un fichier JSON, téléchargez un fichier contenant les valeurs (vous pouvez télécharger un exemple ci-dessous).
    4. Cliquez sur 'Prédire' pour voir si la personne est probablement malade ou saine, avec une explication.
    """)

    # Sélection du mode d'entrée
    mode = st.selectbox("Choisissez le mode d'entrée", ["Saisie manuelle", "Importer un fichier JSON"])

    # Affichage du mode sélectionné
    if mode == "Saisie manuelle":
        display_manual_input(feature_names, scaler, selector, model)
    elif mode == "Importer un fichier JSON":
        display_json_import(feature_names, scaler, selector, model)

    # Affichage de l'importance globale
    display_global_importance(feature_names, scaler, selector, model)

    # Informations supplémentaires
    st.divider()
    st.info("""
    Pour tester avec un fichier JSON, créez un fichier contenant les variables attendues, par exemple :
    ```
    {
      "MDVP:Fo(Hz)": 119.992,
      "MDVP:Fhi(Hz)": 157.302,
      ...
    }
    ```
    Vous pouvez aussi télécharger un exemple en cliquant sur le bouton dans la section 'Importer un fichier JSON'.
    """)

    # Section "À Propos"
    st.markdown("### À Propos")
    st.markdown("""
    Cette application utilise un modèle d'intelligence artificielle (XGBoost) pour prédire la maladie de Parkinson à partir de caractéristiques vocales. Les graphiques SHAP expliquent comment chaque caractéristique influence la prédiction, rendant les résultats plus compréhensibles.
    """)