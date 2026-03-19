import pandas as pd
import pickle

# 1. Load the trained models
with open("models/MM-2026-LogReg.pkl", "rb") as f:
    logreg_model = pickle.load(f)

with open("models/MM-2026-XGBoost.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# 2. Load the preprocessed 2026 submission features
sub_df = pd.read_csv("clean_data/2026_submission_features.csv")

# Extract the IDs for the final Kaggle submission
submission_ids = sub_df["ID"]

# Drop columns that weren't used during training
X_sub = sub_df.drop(columns=["Season", "ID"])

# Handle any missing data 
X_sub = X_sub.fillna(0) 

# 3. Generate Predictions
logreg_preds = logreg_model.predict_proba(X_sub)[:, 1]
xgb_preds = xgb_model.predict_proba(X_sub)[:, 1]

# 4. Ensemble the predictions
final_preds = (logreg_preds + xgb_preds) / 2

# 5. Create the Kaggle submission DataFrame
submission = pd.DataFrame({
    "ID": submission_ids,
    "Pred": final_preds
})

# 6. Save the submission to CSV
submission.to_csv("m_submission.csv", index=False)

print(submission.head())
