import os
import warnings
import logging
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
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
import joblib
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

def convert_numpy_types_to_native(obj):
    """Recursively converts numpy types in a dictionary or list to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_to_native(i) for i in obj]
    return obj

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
    try:
        data = pd.read_csv(url)
        logger.info("Dataset loaded successfully.")
        X = data.drop(columns=['name', 'status'])
        y_original = data['status']
        le = LabelEncoder()
        y = le.fit_transform(y_original.astype(int))
        logger.info("Valeurs uniques dans y (après encodage): %s, Type: %s", np.unique(y), y.dtype)
        logger.info("Classes apprises par LabelEncoder: %s", le.classes_)
        logger.info("Forme de X: %s, Forme de y: %s", X.shape, y.shape)
        return X, y, le
    except Exception as e:
        logger.error("Erreur lors du chargement des données: %s", e)
        raise

def plot_class_distribution(y, le):
    """
    Plot the distribution of classes and save the figure.

    Parameters:
    y (np.array): Target variable (encoded).
    le (LabelEncoder): Label encoder for class names.
    """
    try:
        class_labels = [str(cls) for cls in le.classes_]
        
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(x=y)
        ax.set_xticklabels(class_labels) 
        plt.title('Distribution des Classes')
        plt.xlabel('Statut Parkinson')
        plt.ylabel("Nombre d'individus")
        plt.savefig(os.path.join('figures', 'class_distribution.png'))
        plt.close()
        logger.info("Graphe de distribution des classes sauvegardé.")
    except Exception as e:
        logger.error("Erreur lors de la création du graphe de distribution des classes: %s", e)


def plot_correlation_matrix(X):
    """
    Plot the correlation matrix of features and save the figure.

    Parameters:
    X (pd.DataFrame): Features.
    """
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(X.corr(), cmap='coolwarm', annot=False, fmt=".2f")
        plt.title('Matrice de Corrélation')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', 'correlation_matrix.png'))
        plt.close()
        logger.info("Matrice de corrélation sauvegardée.")
    except Exception as e:
        logger.error("Erreur lors de la création de la matrice de corrélation: %s", e)


def preprocess_data(X, y, k_best=10):
    """
    Preprocess the data: split, scale, apply SMOTE, and select features.
    """
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but SMOTE was fitted with feature names",
        category=UserWarning,
        module='imblearn.base'
    )
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but SelectKBest was fitted with feature names",
        category=UserWarning,
        module='sklearn.feature_selection._univariate_selection'
    )
    warnings.filterwarnings(
        "ignore",
        message="X has feature names, but SelectKBest was fitted without feature names"
    )

    try:
        logger.info("Séparation des données en ensembles d'entraînement et de test...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_original = X_test.copy()

        logger.info("Mise à l'échelle des données...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logger.info("Application de SMOTE pour gérer le déséquilibre des classes...")
        logger.info("Distribution des classes avant SMOTE: %s", np.unique(y_train, return_counts=True))
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        logger.info("Distribution des classes après SMOTE: %s", np.unique(y_train_smote, return_counts=True))

        logger.info("Sélection des %d meilleures features avec mutual_info_classif...", k_best)
        selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
        X_test_selected = selector.transform(X_test_scaled)
        logger.info("Forme des données après sélection: X_train_selected %s, X_test_selected %s", X_train_selected.shape, X_test_selected.shape)

        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = X.columns[selected_feature_indices].tolist()
        logger.info("Features sélectionnées: %s", selected_feature_names)

        return X_train_selected, X_test_selected, y_train_smote, y_test, scaler, selector, X_test_original
    except Exception as e:
        logger.error("Erreur lors du prétraitement des données: %s", e)
        raise
    finally:
        warnings.resetwarnings()


def objective_rf(trial, X_train, y_train):
    """Optuna objective function for Random Forest."""
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold, y_fold = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        if len(np.unique(y_fold)) < 2:
            continue
        try:
            model.fit(X_fold, y_fold)
            score = model.score(X_val, y_val)
            scores.append(score)
        except Exception as e:
             logger.error("Error during RF training on fold %d: %s", fold_idx, e)
             scores.append(np.nan)
    return np.mean(scores) if scores and not np.all(np.isnan(scores)) else -1.0


def objective_xgb(trial, X_train, y_train):
    """Optuna objective function for XGBoost."""
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    gamma = trial.suggest_float('gamma', 0.0, 0.5)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True)

    neg_count, pos_count = np.bincount(y_train.astype(int))
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold, y_fold = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        if len(np.unique(y_fold)) < 2:
            continue
        try:
            model.fit(X_fold, y_fold, eval_set=[(X_val, y_val)], verbose=False)
            score = model.score(X_val, y_val)
            scores.append(score)
        except Exception as e:
             logger.error("Error during XGBoost training on fold %d: %s", fold_idx, e)
             scores.append(np.nan)

    return np.mean(scores) if scores and not np.all(np.isnan(scores)) else -1.0


def train_models(X_train, y_train):
    logger.info("Début de l'optimisation des hyperparamètres avec Optuna...")

    logger.info("Optimisation pour Random Forest...")
    study_rf = optuna.create_study(direction='maximize', study_name='rf_optimization')
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=50, show_progress_bar=True)
    logger.info("Meilleurs hyperparamètres RF: %s", study_rf.best_params)
    model_rf_optimized = RandomForestClassifier(**study_rf.best_params, class_weight='balanced', random_state=42, n_jobs=-1)
    model_rf_optimized.fit(X_train, y_train)
    logger.info("Modèle Random Forest entraîné.")

    logger.info("Optimisation pour XGBoost...")
    study_xgb = optuna.create_study(direction='maximize', study_name='xgb_optimization')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=50, show_progress_bar=True)
    logger.info("Meilleurs hyperparamètres XGBoost: %s", study_xgb.best_params)
    
    neg_count, pos_count = np.bincount(y_train.astype(int))
    scale_pos_weight_final = neg_count / pos_count if pos_count > 0 else 1

    model_xgb_optimized = XGBClassifier(
        **study_xgb.best_params,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_final
    )
    model_xgb_optimized.fit(X_train, y_train)
    logger.info("Modèle XGBoost entraîné.")

    return model_rf_optimized, model_xgb_optimized


def evaluate_model(model, X_test, y_test, model_name, le):
    try:
        logger.info("Évaluation du modèle %s...", model_name)
        y_pred = model.predict(X_test)

        target_names_str = [str(c) for c in le.classes_]
        report_dict = classification_report(y_test, y_pred, target_names=target_names_str, output_dict=True)
        
        report_dict_cleaned = convert_numpy_types_to_native(report_dict)
        
        logger.info("%s - Rapport de classification:\\n%s", model_name, json.dumps(report_dict_cleaned, indent=2))
        accuracy = accuracy_score(y_test, y_pred)
        logger.info("Précision du modèle %s : %.2f%%", model_name, accuracy * 100)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_str, yticklabels=target_names_str)
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.xlabel('Prédit')
        plt.ylabel('Vrai')
        plt.savefig(os.path.join('figures', f'confusion_matrix_{model_name}.png'))
        plt.close()
        logger.info("Matrice de confusion sauvegardée pour %s.", model_name)

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
            logger.info("Courbe ROC sauvegardée pour %s.", model_name)
        else:
             logger.warning("Le modèle %s ne supporte pas predict_proba pour la courbe ROC.", model_name)

    except Exception as e:
        logger.error("Erreur lors de l'évaluation du modèle %s: %s", model_name, e)


def interpret_model(model, X_test_selected, selected_feature_names, model_name="XGBoost"):
    try:
        logger.info("Interprétation du modèle %s avec SHAP...", model_name)
        
        if X_test_selected.shape[0] == 0:
            logger.warning("X_test_selected is empty. Skipping SHAP interpretation for %s.", model_name)
            return
        
        if "XGBoost" in model_name or "RandomForest" in model_name or "LGBM" in model_name:
             explainer = shap.TreeExplainer(model)
        else:
             logger.warning("Modèle %s non pris en charge par TreeExplainer SHAP. Tentative avec KernelExplainer (peut être lent).", model_name)
             background_data = shap.sample(X_test_selected, min(100, X_test_selected.shape[0]))
             explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, background_data)

        X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)
        shap_values = explainer.shap_values(X_test_selected_df)

        if isinstance(shap_values, list):
             if len(shap_values) == 2:
                 shap_values_for_plot = shap_values[1]
             else:
                 logger.warning("SHAP values list has unexpected length %d. Using first element for summary plot.", len(shap_values))
                 shap_values_for_plot = shap_values[0]
        else:
             shap_values_for_plot = shap_values

        if shap_values_for_plot is None or (hasattr(shap_values_for_plot, 'shape') and shap_values_for_plot.shape[0] == 0):
            logger.warning("SHAP values for plotting are empty or invalid for %s. Skipping SHAP summary plot.", model_name)
            return

        plt.figure()
        shap.summary_plot(
            shap_values_for_plot,
            features=X_test_selected_df,
            feature_names=selected_feature_names,
            show=False,
            plot_type='dot'
        )
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join('figures', f'shap_summary_{model_name}.png'), bbox_inches='tight')
        plt.close()
        logger.info("SHAP summary plot sauvegardé pour %s.", model_name)

    except Exception as e:
        logger.error("Erreur lors de l'interprétation SHAP pour le modèle %s: %s", model_name, e)


def run_internal_tests(model, X_test_original, y_test, scaler, selector, original_feature_names, le):
    logger.info("Exécution des tests internes sur des échantillons aléatoires du jeu de test...")
    try:
        num_samples = min(len(X_test_original), 10)
        if num_samples == 0:
            logger.warning("Jeu de test original vide, tests internes sautés.")
            return
        
        sample_indices = np.random.choice(X_test_original.index, num_samples, replace=False)

        for i, idx in enumerate(sample_indices):
            sample_original_df = X_test_original.loc[[idx]]
            true_label_idx_in_y_test = X_test_original.index.get_loc(idx)
            true_label_encoded = y_test[true_label_idx_in_y_test]
            true_label_name = str(le.inverse_transform([true_label_encoded])[0])

            sample_scaled = scaler.transform(sample_original_df)
            sample_selected = selector.transform(sample_scaled)

            prediction_encoded = model.predict(sample_selected)[0]
            prediction_name = str(le.inverse_transform([prediction_encoded])[0])

            if hasattr(model, 'predict_proba'):
                proba_all_classes = model.predict_proba(sample_selected)[0]
                probability = proba_all_classes[prediction_encoded] 
                logger.info("Échantillon %d (Index %d): Prédit '%s' (Prob %.2f), Vrai '%s'",
                            i + 1, idx, prediction_name, probability, true_label_name)
            else:
                logger.info("Échantillon %d (Index %d): Prédit '%s', Vrai '%s'",
                            i + 1, idx, prediction_name, true_label_name)

        logger.info("Tests internes terminés.")
    except Exception as e:
        logger.error("Erreur lors de l'exécution des tests internes: %s", e)


def save_assets(model_xgb, model_rf, scaler, selector, original_feature_names):
    try:
        joblib.dump(model_xgb, os.path.join('models', 'model_xgb.pkl'))
        joblib.dump(model_rf, os.path.join('models', 'model_rf.pkl'))
        joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
        joblib.dump(selector, os.path.join('models', 'selector.pkl'))
        with open(os.path.join('models', 'feature_names.json'), 'w') as f:
            json.dump(original_feature_names, f)
        logger.info("Modèles et objets de prétraitement sauvegardés dans le répertoire 'models'.")
    except Exception as e:
        logger.error("Erreur lors de la sauvegarde des modèles et objets: %s", e)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Parameters: { \"use_label_encoder\" } are not used.", category=UserWarning, module="xgboost.training")
    
    warnings.filterwarnings("ignore", message="Using categorical units to plot a list of strings that are all parsable as floats or dates.", category=UserWarning, module="matplotlib.category")


    logger.info("Début du script d'entraînement...")

    DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
    K_BEST_FEATURES = 10

    X_original, y_encoded, label_encoder = load_data(DATA_URL)
    original_feature_names_list = X_original.columns.tolist()

    plot_class_distribution(y_encoded, label_encoder)
    plot_correlation_matrix(X_original)

    X_train_processed, X_test_processed, y_train_processed, y_test_processed, \
        data_scaler, feature_selector, X_test_original_df = preprocess_data(
            X_original, y_encoded, k_best=K_BEST_FEATURES
    )

    model_rf_optimized, model_xgb_optimized = train_models(X_train_processed, y_train_processed)

    evaluate_model(model_rf_optimized, X_test_processed, y_test_processed, 'RandomForest', label_encoder)
    evaluate_model(model_xgb_optimized, X_test_processed, y_test_processed, 'XGBoost', label_encoder)

    selected_feature_indices = feature_selector.get_support(indices=True)
    selected_feature_names_list = X_original.columns[selected_feature_indices].tolist()
    interpret_model(model_xgb_optimized, X_test_processed, selected_feature_names_list, model_name='XGBoost')

    run_internal_tests(model_xgb_optimized, X_test_original_df, y_test_processed, 
                       data_scaler, feature_selector, original_feature_names_list, label_encoder)

    save_assets(model_xgb_optimized, model_rf_optimized, data_scaler, feature_selector, original_feature_names_list)

    logger.info("Script d'entraînement terminé. Modèles et assets sauvegardés.")
    
    warnings.resetwarnings()