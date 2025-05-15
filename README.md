# Parkinson's Disease Prediction Project

This project uses machine learning to predict Parkinson's disease based on vocal features, such as voice frequency and amplitude. It includes a user-friendly web interface built with Streamlit and a backend server powered by Flask.

## Table of Contents

- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [Usage](#usage)

## Installation

Follow these steps to set up the project on your computer:

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. **Create a Virtual Environment** (recommended to keep dependencies isolated):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If you encounter version conflicts, consider specifying versions by running `pip freeze > requirements.txt` in your working environment and sharing that file.

## Running the Application

To run the application locally:

1. **Verify Required Files**:
   - Ensure the following files are in the same directory as `interface_streamlit.py`:
     - `model_xgb.pkl` (machine learning model)
     - `scaler.pkl` (data scaler)
     - `selector.pkl` (feature selector)
     - `feature_names.json` (list of feature names)

2. **Launch the Streamlit Interface**:
   ```bash
   streamlit run interface_streamlit.py
   ```

3. **Access the Application**:
   - Open your web browser and go to `http://localhost:8501`.

## Deployment

To share your application online for your PFE presentation, you can deploy it to [Heroku](https://www.heroku.com/), a platform that simplifies hosting Python applications.

### Deploying to Heroku

1. **Create a Heroku Account**:
   - Sign up at [Heroku](https://www.heroku.com/).

2. **Install Heroku CLI**:
   - Download and install the Heroku Command Line Interface from [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).

3. **Create a Procfile**:
   - In your project directory, create a file named `Procfile` with:
     ```
     web: streamlit run interface_streamlit.py --server.port $PORT --server.address 0.0.0.0
     ```

4. **Commit Your Code**:
   - Initialize a Git repository if not already done, and commit all files:
     ```bash
     git init
     git add .
     git commit -m "Initial commit"
     ```

5. **Deploy to Heroku**:
   - Create a Heroku app:
     ```bash
     heroku create
     ```
   - Push your code to Heroku:
     ```bash
     git push heroku main
     ```
   - Open the deployed application:
     ```bash
     heroku open
     ```

   **Note**: Replace `main` with your branch name if different (e.g., `master`).

## Usage

The application offers two ways to input data for prediction:

- **Manual Input**:
  - Choose "Saisie manuelle" from the dropdown menu.
  - Enter values for vocal features (e.g., voice frequency in Hz).
  - Click "Prédire" to see if the person is likely "Malade" (has Parkinson's) or "Sain" (healthy), along with a probability score.
  - A SHAP graph explains which features influenced the prediction:
    - **Red bars** suggest a higher chance of Parkinson's.
    - **Blue bars** suggest a lower chance.
    - Longer bars mean the feature had a bigger impact.

- **JSON Import**:
  - Choose "Importer un fichier JSON".
  - Upload a JSON file with feature names and values (e.g., `{"MDVP:Fo(Hz)": 119.992}`).
  - Download an example JSON file from the interface if needed.
  - Click "Prédire" to view the prediction and SHAP explanation.

- **Global Feature Importance**:
  - At the bottom, a SHAP summary plot shows which vocal features are most important overall for predicting Parkinson's, helping you understand what matters most.
