
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Small training set
data = pd.DataFrame({
    "hours": [1,2,3,4,5,6,7,8,9],
    "passed": [0,0,0,0,1,1,1,1,1]
})

X = data[["hours"]]
y = data["passed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Trained. Test accuracy: {acc:.2f}")

joblib.dump(model, "model.joblib")
print("Saved model.joblib")
