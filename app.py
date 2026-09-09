"""
Customer Transaction Prediction — Streamlit demo app (v2)
-----------------------------------------------------------
WHY THIS VERSION IS DIFFERENT FROM THE ORIGINAL:

The original version exposed only the top-10 most important features as
manual inputs and filled the other 188 with training-data medians. Testing
against known-label rows showed this loses too much signal for this
dataset — a real "Transaction" row scored 0.8037 probability using its
full 198 real features, but only 0.1381 through the app using top-10 +
medians. The model itself is correct; the simplification wasn't.

This version instead lets the user pick a REAL customer record (with all
198 real feature values) from a small bundled sample of the test set, so
every prediction reflects what the model can actually do. It also shows
the true label alongside the prediction, which doubles as a live,
honest accuracy demonstration for anyone viewing this app.

Loads:
    - customer_transaction_prediction.pkl -> CustTransPredEnsemble
    - scaler.pkl                          -> fitted scaler (raw-space)
    - feature_columns.pkl                 -> exact column order
    - sample_test_rows.csv                -> a small set of REAL test-set
                                              rows (all 198 features +
                                              true_label), exported from
                                              the training notebook
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ------------------------------------------------------------------
# Custom ensemble class — MUST be defined here, identically to how it
# was defined in the training notebook, before joblib.load() runs.
# ------------------------------------------------------------------
class CustTransPredEnsemble:
    def __init__(self, xgb, lgbm, w_xgb=0.75):
        self.xgb = xgb
        self.lgbm = lgbm
        self.w_xgb = w_xgb
        self.w_lgbm = 1 - w_xgb

    def predict_proba(self, X):
        p1 = self.xgb.predict_proba(X)[:, 1]
        p2 = self.lgbm.predict_proba(X)[:, 1]
        final_prob = (self.w_xgb * p1) + (self.w_lgbm * p2)
        return np.vstack([1 - final_prob, final_prob]).T

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = "customer_transaction_prediction.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
SAMPLE_ROWS_PATH = "sample_test_rows.csv"

# From the notebook: blend_threshold = (w_xgb * xgb_best_threshold) +
# (w_lgbm * lgbm_best_threshold) = 0.6148 (rounded). Using the tuned
# threshold rather than the default 0.5, matching what the notebook's
# own evaluation used.
DECISION_THRESHOLD = 0.6148


# ------------------------------------------------------------------
# Cached artifact loading
# ------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns


@st.cache_data
def load_sample_rows():
    return pd.read_csv(SAMPLE_ROWS_PATH)


# ------------------------------------------------------------------
# App UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Customer Transaction Prediction", layout="centered")
st.title("Customer Transaction Prediction")
st.caption(
    "Select a real customer record from the test set below. The model "
    "predicts using that record's complete, real feature data — not "
    "manually typed or estimated values."
)

model, scaler, feature_columns = load_model_artifacts()
sample_df = load_sample_rows()

st.divider()
st.subheader("Choose a Customer Record")

record_id = st.selectbox("Record", sample_df["record_id"].tolist())
selected_row = sample_df[sample_df["record_id"] == record_id].iloc[0]

with st.expander("View this record's raw feature values"):
    st.dataframe(
        selected_row[feature_columns].to_frame(name="value"),
        use_container_width=True,
    )

st.divider()

if st.button("Predict", type="primary"):
    with st.spinner("Running inference..."):
        raw_features = selected_row[feature_columns].to_frame().T
        scaled_features = scaler.transform(raw_features)

        proba = model.predict_proba(scaled_features)[0][1]
        pred_class = int(proba >= DECISION_THRESHOLD)

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Predicted Class",
            "Transaction (1)" if pred_class == 1 else "No Transaction (0)",
        )
    with col2:
        st.metric("Probability", f"{proba:.2%}")
    with col3:
        true_label = int(selected_row["true_label"])
        st.metric(
            "Actual Label",
            "Transaction (1)" if true_label == 1 else "No Transaction (0)",
        )

    if pred_class == true_label:
        st.success("Prediction matches the actual outcome.")
    else:
        st.warning("Prediction did not match the actual outcome for this record.")

    st.caption(
        "This model achieves ~0.88 test ROC-AUC. For the minority "
        "'Transaction' class specifically, it has ~50% precision and "
        "~60% recall — meaning it correctly identifies about 6 in 10 "
        "real transactions, and roughly half of its 'Transaction' "
        "predictions are correct. Occasional misses are expected and "
        "consistent with that measured performance."
    )
