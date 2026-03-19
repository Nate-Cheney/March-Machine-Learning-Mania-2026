import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# 1. Load preprocessed features
feature_df = pd.read_csv("clean_data/w_features.csv")

# 2. Prepare for training
# Train on 2003 through 2022
train_df = feature_df[(feature_df["Season"] >= 2003) & (feature_df["Season"] <= 2022)]

# Test on 2024 and 2025
test_df = feature_df[feature_df["Season"].isin([2024, 2025])]

X_train = train_df.drop(columns=["TeamA_Won", "Season"])
y_train = train_df["TeamA_Won"]

X_test = test_df.drop(columns=["TeamA_Won", "Season"])
y_test = test_df["TeamA_Won"]

# 3. Train model(s)
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

with open(f"models/WM-2026-LogReg.pkl", "wb") as f:
    pickle.dump(logreg_model, f)

xgb_model = XGBClassifier(
    objective="binary:logistic",
    learning_rate=0.01,
    n_estimators=1800,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

with open(f"models/WM-2026-XGBoost.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# 4. Test model
y_pred = logreg_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

y_pred = xgb_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

