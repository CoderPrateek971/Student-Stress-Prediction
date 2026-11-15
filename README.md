# Student Stress Predictor

A Streamlit app that uses the selected KNN pipeline from `Final_Dataset_Ml_project.ipynb` to predict Low, Medium, or High stress from ten questionnaire responses.

## Add the trained model

The trained pipeline is saved as `Best_KNN_Pipeline.joblib` in this project folder beside `app.py`. If you need to recreate it, run the notebook's final pipeline saving cell in Colab and download it from `MyDrive/ML_Project_Artifacts`.

The app sends numeric scores (Never = 1 through Very Often = 5) to the saved pipeline in the training feature order. The pipeline should include its fitted imputer/scaler and KNN classifier; no retraining is done by the app.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The saved pipeline was created with scikit-learn 1.6.1, so `requirements.txt` pins that version. Scikit-learn does not guarantee compatibility when loading persisted models across versions.


