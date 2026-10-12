from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

FEATS = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES",
         "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]

SEGMENTS = {
    "Cash-Advance Dependent": "Takes cash advances, keeps purchases and payments low.",
    "Low-Activity / Low-Balance": "Low balances and little activity — dormant or barely used card.",
    "Premium Active Spenders": "High spenders with high limits; frequent one-off purchases and solid payments.",
}

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_pipeline():
    return joblib.load(ROOT / "models" / "pipeline.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "processed" / "df_clean_clustered.csv")


pipeline = load_pipeline()
df = load_data()

st.title("CardProfiler — customer segment demo")
st.caption("Uses models/pipeline.joblib as-is, scaler included — raw values go straight in.")

df["label"] = "Customer #" + df.index.astype(str)

option = st.selectbox(
    "Start from a real customer, or enter a profile manually",
    ["Enter manually"] + df["label"].tolist(),
)

defaults = df[FEATS].median()
if option != "Enter manually":
    defaults = df.loc[df["label"] == option, FEATS].iloc[0]

values = {}
left, right = st.columns(2)
for i, feat in enumerate(FEATS):
    column = left if i % 2 == 0 else right
    with column:
        values[feat] = st.number_input(
            feat, min_value=0.0, value=float(defaults[feat]), step=100.0)

if st.button("Predict segment"):
    row = pd.DataFrame([values])[FEATS]

    segment = pipeline.predict(row)[0]
    proba = pipeline.predict_proba(row)[0]

    st.subheader(segment)
    st.write(SEGMENTS[segment])

    probabilities = pd.DataFrame(
        {"segment": pipeline.classes_, "probability": proba})
    st.write("Confidence per segment")
    st.write(probabilities.round(4))
    st.bar_chart(pd.DataFrame({"probability": proba}, index=pipeline.classes_))