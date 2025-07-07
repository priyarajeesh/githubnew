import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
data = {
    "Income": [50000,60000,25000,70000,80000],
    "CreditScore":[700,650,600,780,500],
    "LoanAmount":[200000, 300000, 250000, 200000, 1000000],
    "LoanTerm":[360,360,500,170,240],
    "Approved":["Yes","Yes","No","No","Yes"]
}
df = pd.DataFrame(data)
le = LabelEncoder()
df['Approved'] = le.fit_transform(df["Approved"])
X = df[["Income", "CreditScore", "LoanAmount", "LoanTerm"]]
y= df["Approved"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
model_path = "./model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model,f)