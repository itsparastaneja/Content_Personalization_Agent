"""STEP 2: Train the click-prediction model.
Run:  python train_model.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "data" / "impressions.csv")

# Feature 1: how often THIS user clicks THIS category (not counting the row itself)
g = df.groupby(["user_id", "category"])["clicked"]
clicks, shown = g.transform("sum"), g.transform("count")
df["user_cat_rate"] = (clicks - df["clicked"] + 1) / (shown - 1 + 2)

# Feature 2: how popular this content is with everybody (not counting the row itself)
p = df.groupby("content_id")["clicked"]
pclicks, pshown = p.transform("sum"), p.transform("count")
df["popularity"] = (pclicks - df["clicked"] + 1) / (pshown - 1 + 2)

X = df[["user_cat_rate", "popularity"]]
y = df["clicked"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy on unseen rows:", round(model.score(X_test, y_test), 3))
print("Weights (bigger = more important):")
print("  user_cat_rate:", round(model.coef_[0][0], 2))
print("  popularity   :", round(model.coef_[0][1], 2))

joblib.dump(model, BASE / "model.joblib")
print("Saved model.joblib")
