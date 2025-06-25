import streamlit as st
import joblib
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import warnings

warnings.filterwarnings(
    "ignore",
    message="X has feature names, but RandomForestClassifier was fitted without feature names",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but XGBClassifier was fitted without feature names",
    category=UserWarning
)

st.set_page_config(page_title="Parkinson Predictor", layout="wide", page_icon="🩺")

CONFIG_PATH = Path("config.json")
MODELS_DIR = Path("models")

MODEL_FILES = {
    "XGBoost": {"file": "model_xgb.pkl", "class": XGBClassifier},
    "Random Forest": {"file": "model_rf.pkl", "class": RandomForestClassifier},
}

FEATURE_DESCRIPTIONS = {
    "Jitter(%)": "Variation relative de la fréquence fondamentale (Jitter %)",
    "Jitter(Abs)": "Variation absolue de la fréquence fondamentale (Jitter Abs)",
    "Jitter:RAP": "Relative Average Perturbation (RAP)",
    "Jitter:PPQ5": "Pitch Period Perturbation Quotient (PPQ5)",
    "Jitter:DDP": "Difference of Differences of Periods (DDP)",
    "Shimmer": "Variation relative de l'amplitude (Shimmer)",
    "Shimmer(dB)": "Variation de l'amplitude en décibels (Shimmer dB)",
    "Shimmer:APQ3": "Amplitude Perturbation Quotient (3 cycles)",
    "Shimmer:APQ5": "Amplitude Perturbation Quotient (5 cycles)",
    "Shimmer:APQ11": "Amplitude Perturbation Quotient (11 cycles)",
    "Shimmer:DDA": "Difference of Differences of Amplitude (DDA)",
    "NHR": "Noise-to-Harmonics Ratio (NHR)",
    "HNR": "Harmonics-to-Noise Ratio (HNR)",
    "RPDE": "Recurrence Period Density Entropy (RPDE)",
    "DFA": "Detrended Fluctuation Analysis (DFA)",
    "PPE": "Pitch Period Entropy (PPE)"
}


@st.cache_resource
def load_common_assets():
    scaler_path = MODELS_DIR / 'scaler.pkl'
    selector_path = MODELS_DIR / 'selector.pkl'
    features_path = MODELS_DIR / 'feature_names.json'
    if not scaler_path.exists() or not selector_path.exists() or not features_path.exists():
        st.error(f"Actifs communs (scaler, selector, feature_names) non trouvés dans '{MODELS_DIR}'. Exécutez train.py.")
        return None, None, None
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    return scaler, selector, feature_names

@st.cache_resource
def load_predictive_model(model_key: str):
    if model_key not in MODEL_FILES:
        st.error(f"Clé de modèle '{model_key}' non reconnue.")
        return None
    
    model_filename = MODEL_FILES[model_key]["file"]
    model_path = MODELS_DIR / model_filename
    
    if not model_path.exists():
        st.error(f"Fichier modèle '{model_filename}' non trouvé dans '{MODELS_DIR}'. A-t-il été entraîné et sauvegardé ?")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle '{model_key}' depuis '{model_path}': {e}")
        return None


def app():
    st.title("🔍 Dépistage Précoce de la Maladie de Parkinson")
    st.markdown("---Saisissez les paramètres vocaux ou importez un fichier JSON pour obtenir une prédiction.---")

    common_scaler, common_selector_trained, all_feature_names = load_common_assets()

    if not all([common_scaler, common_selector_trained, all_feature_names]):
        st.error("Impossible de charger les actifs de prétraitement essentiels. L'application ne peut pas continuer.")
        st.stop()

    k_model_was_trained_with = common_selector_trained.k if hasattr(common_selector_trained, 'k') else \
                               (len(common_selector_trained.get_support(indices=True)) if hasattr(common_selector_trained, 'get_support') else 10)


    with st.sidebar:
        st.header("Guide d'Utilisation")
        with st.expander("📖 Comment utiliser cette application ?", expanded=False):
            st.markdown("""
            Bienvenue ! Cette application vous aide à évaluer le risque de maladie de Parkinson à partir de données vocales.
            
            **Suivez ces 4 étapes simples :**

            **1. Configurez votre analyse (Optionnel)**
            *   Dans la section `Configuration de l'Analyse` ci-dessous, vous pouvez choisir le modèle d'intelligence artificielle à utiliser (`XGBoost` est recommandé).
            
            **2. Saisissez les données du patient**
            *   **Méthode 1 : Saisie manuelle**
                - Cochez l'option `📝 Saisie manuelle`.
                - Remplissez les 22 champs avec les mesures vocales du patient.
            *   **Méthode 2 : Importer un fichier**
                - Cochez l'option `📤 Importer un fichier JSON`.
                - Cliquez sur `Browse files` et sélectionnez le fichier JSON contenant les données.
            
            **3. Lancez l'analyse**
            *   Cliquez sur le grand bouton bleu `🔬 Analyser et Prédire le Risque`.

            **4. Interprétez les résultats**
            *   **Le diagnostic :** Un message clair (**Risque Élevé** ou **Faible Risque**) apparaîtra.
            *   **Les explications (Graphique SHAP) :** Pour comprendre le *pourquoi* :
                - Les barres **<font color='red'>rouges</font>** montrent les facteurs qui ont **augmenté** le risque.
                - Les barres **<font color='green'>vertes</font>** montrent les facteurs qui ont **réduit** le risque.
                - Plus une barre est longue, plus son influence est forte.

            ---
            **Avertissement Important :**
            Cet outil est une aide au dépistage et ne remplace **en aucun cas** un diagnostic médical posé par un professionnel de santé qualifié.
            """, unsafe_allow_html=True)

        st.header("⚙️ Configuration de l'Analyse")
        default_app_config = {"selected_model_key": "XGBoost", 
                              "k_best_for_shap": k_model_was_trained_with, 
                              "show_shap": True}
        app_config = default_app_config.copy()
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r") as f:
                    loaded_config = json.load(f)
                    for key, default_value in default_app_config.items():
                        app_config[key] = loaded_config.get(key, default_value)
            except (json.JSONDecodeError, Exception) as e:
                st.warning(f"Erreur chargement {CONFIG_PATH}: {e}. Config par défaut utilisée.")
        
        available_model_keys = list(MODEL_FILES.keys())
        selected_model_key = st.selectbox(
            "Choisissez le modèle de prédiction :",
            options=available_model_keys,
            index=available_model_keys.index(app_config["selected_model_key"]) if app_config["selected_model_key"] in available_model_keys else 0,
            key="sb_model_select"
        )

        max_k_for_shap = min(k_model_was_trained_with, len(all_feature_names)) 
        k_best_for_shap_value = st.slider(
            f"Nombre de features à afficher dans SHAP (max {max_k_for_shap}) :",
            min_value=1,
            max_value=max_k_for_shap,
            value=min(app_config.get("k_best_for_shap", max_k_for_shap), max_k_for_shap),
            key="sl_k_best_shap",
            help=f"Le modèle a été entraîné avec {k_model_was_trained_with} features. Vous pouvez choisir d'en afficher moins dans les explications SHAP."
        )

        show_shap_flag = st.checkbox(
            "Afficher les explications SHAP détaillées",
            value=app_config.get("show_shap", True),
            key="cb_show_shap"
        )

        if st.button("💾 Sauvegarder la Configuration", key="bt_save_config"):
            current_config_to_save = {
                "selected_model_key": selected_model_key,
                "k_best_for_shap": k_best_for_shap_value, 
                "show_shap": show_shap_flag
            }
            try:
                with open(CONFIG_PATH, "w") as f:
                    json.dump(current_config_to_save, f, indent=4)
                st.sidebar.success("Configuration sauvegardée !")
                app_config.update(current_config_to_save)
            except Exception as e:
                 st.sidebar.error(f"Erreur sauvegarde config : {e}")

        st.sidebar.button("🔄 Réinitialiser le formulaire", on_click=lambda: st.session_state.clear())
        st.sidebar.download_button(
            label="⬇️ Télécharger un exemple JSON",
            data='{"MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, ...}',
            file_name="exemple_patient.json",
            mime="application/json"
        )

    st.subheader("⬇️ Entrée des Données Utilisateur")
    input_method = st.radio(
        "Choisissez la méthode d'entrée des données :",
        ["📝 Saisie manuelle", "📤 Importer un fichier JSON"],
        horizontal=True, key="rg_input_method"
    )

    input_df = None
    default_manual_input_value = 0.0 

    if "Saisie manuelle" in input_method:
        input_df = manual_input_form(all_feature_names, default_value=default_manual_input_value)
    else:
        input_df = file_upload_form(all_feature_names)

    st.markdown("---")

    if st.button("🔬 Analyser et Prédire le Risque", type="primary", key="bt_predict"):
        proceed_to_predict = False
        if input_df is not None:
            if "Saisie manuelle" in input_method:
                if (input_df == default_manual_input_value).all().all():
                    st.warning("Veuillez saisir des valeurs pour les paramètres vocaux ou importer un fichier. "
                               "Tous les champs manuels ont leur valeur par défaut.")
                    st.session_state.display_results = False
                else:
                    proceed_to_predict = True
            else:
                proceed_to_predict = True 
        else:
            st.warning("Veuillez saisir ou importer des données valides avant de lancer l'analyse.")
            st.session_state.display_results = False

        if proceed_to_predict:
            st.session_state.prediction_params = {
                "input_df": input_df,
                "model_key": selected_model_key,
                "k_best_for_shap": k_best_for_shap_value, 
                "show_shap": show_shap_flag,
                "scaler": common_scaler,
                "selector_trained": common_selector_trained, 
                "all_feature_names": all_feature_names,
                "k_model_trained_with": k_model_was_trained_with
            }
            st.session_state.display_results = True

    if st.session_state.get('display_results', False) and 'prediction_params' in st.session_state:
        st.subheader("📊 Résultats de l'Analyse")
        display_prediction_results(**st.session_state.prediction_params)


def manual_input_form(feature_names_list, default_value=0.0):
    user_input = {}
    with st.expander("📝 Saisir les 22 paramètres vocaux manuellement", expanded=True):
        cols = st.columns(3)
        for i, feat_name in enumerate(feature_names_list):
            with cols[i % 3]:
                user_input[feat_name] = st.number_input(
                    label=FEATURE_DESCRIPTIONS.get(feat_name, feat_name),
                    min_value=-1000.0, 
                    max_value=1000.0,  
                    value=default_value,    
                    step=0.0001,
                    format="%.4f",
                    key=f"manual_input_{feat_name}", 
                    help=FEATURE_DESCRIPTIONS.get(feat_name, "Valeur numérique attendue.")
                )
    return pd.DataFrame([user_input], columns=feature_names_list)

def file_upload_form(feature_names_list):
    uploaded_file = st.file_uploader(
        "Importer un fichier JSON avec les valeurs des features :",
        type=["json"], key="file_uploader_widget"
    )
    if uploaded_file:
        try:
            data_from_file = json.load(uploaded_file)
            if not isinstance(data_from_file, dict):
                st.error("Le contenu du fichier JSON doit être un objet unique (dictionnaire).")
                return None
            
            input_dict = {}
            for feat in feature_names_list:
                val = data_from_file.get(feat, 0.0)
                try:
                    input_dict[feat] = float(val)
                except (ValueError, TypeError):
                    st.warning(f"Valeur non-numérique '{val}' pour '{feat}' dans le fichier. Remplacé par 0.0.")
                    input_dict[feat] = 0.0
            
            df = pd.DataFrame([input_dict], columns=feature_names_list)
            st.success("Fichier JSON chargé et validé.")
            return df
        except json.JSONDecodeError:
            st.error("Format JSON invalide. Veuillez vérifier le fichier.")
        except Exception as e:
            st.error(f"Erreur de lecture ou de traitement du fichier : {str(e)}")
    return None

def display_prediction_results(input_df, model_key, k_best_for_shap, show_shap, scaler, 
                               selector_trained, all_feature_names, k_model_trained_with):
    try:
        model_to_use = load_predictive_model(model_key)
        if not model_to_use:
            st.error(f"Le chargement du modèle '{model_key}' a échoué. Impossible de prédire.")
            return

        st.info(f"Analyse avec le modèle : **{model_key}**. "
                f"Ce modèle a été entraîné sur **{k_model_trained_with}** features. "
                f"Les explications SHAP afficheront jusqu'à **{k_best_for_shap}** de ces features.")

        X_scaled = scaler.transform(input_df)
        X_selected_for_model_prediction = selector_trained.transform(X_scaled) 

        model_feature_mask = selector_trained.get_support()
        
        if not np.any(model_feature_mask):
            st.warning(f"Aucune feature n'a été sélectionnée par le sélecteur de base du modèle.")
            if (input_df == 0).all().all():
                 st.info("Note : Toutes les valeurs d'entrée étaient à zéro. Cela peut influencer la prédiction.")
            return

        feature_names_for_model = [name for i, name in enumerate(all_feature_names) if model_feature_mask[i]]
        
        if X_selected_for_model_prediction.shape[1] != len(feature_names_for_model):
            st.error(f"Incohérence de forme après transformation: "
                       f"{X_selected_for_model_prediction.shape[1]} features vs "
                       f"{len(feature_names_for_model)} noms. "
                       f"Sélecteur k={selector_trained.k if hasattr(selector_trained, 'k') else 'N/A'}.")
            return

        X_selected_df_for_model = pd.DataFrame(X_selected_for_model_prediction, columns=feature_names_for_model)

        pred_encoded = model_to_use.predict(X_selected_df_for_model)[0]
        
        proba_parkinsons = -1.0 
        if hasattr(model_to_use, 'predict_proba'):
            proba_all = model_to_use.predict_proba(X_selected_df_for_model.values)[0]
            proba_parkinsons = proba_all[1] if len(proba_all) > 1 else proba_all[0]

        status_text = "Risque Élevé de Parkinson Détecté" if pred_encoded == 1 else "Profil Vocal Normal / Faible Risque"
        confidence_text = f"(Probabilité de Parkinson selon le modèle : {proba_parkinsons:.1%})" if proba_parkinsons >= 0 else ""

        if pred_encoded == 1:
            st.error(f"## 🚨 {status_text} {confidence_text}")
        else:
            st.success(f"## ✅ {status_text} {confidence_text}")

        if show_shap and X_selected_df_for_model.shape[1] > 0 :
            with st.expander("🔍 Explications Détaillées de la Prédiction (SHAP)", expanded=True):
                explain_prediction_shap(model_to_use, X_selected_df_for_model, feature_names_for_model, k_best_for_shap)
        elif show_shap and X_selected_df_for_model.shape[1] == 0:
             st.warning("Les explications SHAP ne peuvent pas être affichées car aucune feature n'a été utilisée par le modèle.")

    except Exception as e:
        st.error(f"Une erreur est survenue lors du processus de prédiction : {str(e)}")
        st.exception(e) 


def explain_prediction_shap(model, X_selected_df_single_row, model_features_names, k_best_for_display):
    st.subheader("📊 Influence des variables sur cette prédiction")
    try:
        explainer = shap.Explainer(model)
        shap_explanation = explainer(X_selected_df_single_row)

        # Correction : gérer la forme de shap_explanation.values
        values = shap_explanation.values
        # Cas classification binaire (2D) : (1, n_features)
        if values.ndim == 2:
            shap_values_for_plot = values[0]
        # Cas multi-classes (3D) : (1, n_features, n_classes)
        elif values.ndim == 3:
            shap_values_for_plot = values[0, :, 1]  # classe positive
        else:
            st.error(f"Format inattendu des valeurs SHAP : shape={values.shape}")
            return

        if len(shap_values_for_plot) != len(model_features_names):
            st.error(f"Erreur critique de dimension SHAP : {len(shap_values_for_plot)} valeurs SHAP pour {len(model_features_names)} features.")
            return

        shap_df = pd.DataFrame({
            'Feature': model_features_names,
            'SHAP Value': shap_values_for_plot,
            'Input Value': X_selected_df_single_row.iloc[0].values 
        })
        
        shap_df['abs_SHAP'] = shap_df['SHAP Value'].abs()
        shap_df_sorted_for_bar = shap_df.sort_values(by='abs_SHAP', ascending=True) 
        
        shap_df_to_display_bar = shap_df_sorted_for_bar.tail(k_best_for_display)

        fig_bar = go.Figure()
        colors = ['#ff6961' if val > 0 else '#77dd77' for val in shap_df_to_display_bar['SHAP Value']]
        
        fig_bar.add_trace(go.Bar(
            x=shap_df_to_display_bar['SHAP Value'],
            y=shap_df_to_display_bar['Feature'],
            orientation='h',
            marker_color=colors,
            text=[f"<b>{feat}</b><br>SHAP: {val:.3f}<br>Entrée: {in_val:.4f}" 
                  for feat, val, in_val in zip(shap_df_to_display_bar['Feature'], shap_df_to_display_bar['SHAP Value'], shap_df_to_display_bar['Input Value'])],
            hoverinfo='text'
        ))
        fig_bar.update_layout(
            title_text=f"Impact des {k_best_for_display} Principales Variables sur la Prédiction",
            xaxis_title="Valeur SHAP (Contribution à la prédiction)",
            yaxis_title="Variable",
            height=max(400, k_best_for_display * 40), 
            margin=dict(l=230, r=20, t=60, b=50) 
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown(f"#### Facteurs Clés Ayant Influencé Cette Prédiction (Top {min(3, k_best_for_display)}) :")
        shap_df_sorted_for_text = shap_df.sort_values(by='abs_SHAP', ascending=False)
        for _, row in shap_df_sorted_for_text.head(min(3, k_best_for_display)).iterrows():
            value_contrib_text = f"(valeur entrée : {row['Input Value']:.4f})"
            if row['SHAP Value'] > 0.01:
                direction_text = "a **fortement augmenté** la probabilité de détection de Parkinson"
            elif row['SHAP Value'] < -0.01:
                direction_text = "a **réduit** la probabilité de détection de Parkinson"
            else:
                direction_text = "a eu un impact négligeable"
            
            st.markdown(f"- La variable **{row['Feature']}** {value_contrib_text} {direction_text} (Impact SHAP : {row['SHAP Value']:.3f}).")
        
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'explication SHAP : {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    if 'display_results' not in st.session_state:
        st.session_state.display_results = False
    if 'prediction_params' not in st.session_state:
        st.session_state.prediction_params = {}
    
    app()