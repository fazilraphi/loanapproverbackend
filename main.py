from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://loanfrontend-three.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Input Schema
# -----------------------------
class LoanApplicant(BaseModel):
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str



def classify_risk(prob):
    if prob >= 0.75:
        return "Low Risk"
    elif prob >= 0.5:
        return "Medium Risk"
    else:
        return "High Risk"



@app.post("/predict")
def predict(applicant: LoanApplicant):

    data = applicant.dict()
    df = pd.DataFrame([data])

 
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Estimated_EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["DTI"] = df["Estimated_EMI"] / df["Total_Income"]

    df = pd.get_dummies(df)

   
    df = df.reindex(columns=feature_columns, fill_value=0)

   
    scaled = scaler.transform(df)

  
    probability = model.predict_proba(scaled)[0][1]

    
    risk = classify_risk(probability)

  
    safe_emi = 0.4 * df["Total_Income"].values[0]

    return {
        "approval_probability": round(float(probability), 4),
        "risk_level": risk,
        "safe_emi": round(float(safe_emi), 2)
    }