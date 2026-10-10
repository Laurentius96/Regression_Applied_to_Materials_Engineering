"""Streamlit app: explore the concrete data, compare models and simulate strength.

Run from this folder after executing the notebook:  streamlit run app.py
"""
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
TARGET = "Concrete compressive strength"

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱", layout="wide")


@st.cache_data
def load_data():
    with open(HERE / "dashboard_data.pkl", "rb") as fh:
        return pickle.load(fh)


@st.cache_resource
def load_models():
    with open(HERE / "models.pkl", "rb") as fh:
        return pickle.load(fh)


data = load_data()
models = load_models()
results = data["results_df"]
best = results.set_index("Model").loc[models["best_model_name"]]

st.sidebar.title("🧱 Concrete Strength")
page = st.sidebar.radio("Page", ["Overview", "Data", "Models", "Feature importance", "Simulator"])
st.sidebar.caption("Gradient Boosting and five other regressors trained on the DNC challenge version of the UCI Concrete Compressive Strength data.")

if page == "Overview":
    st.title("Concrete Compressive Strength Prediction")
    st.write("Predict the strength of a concrete mix (MPa) from its components and age.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{len(data['df']):,}")
    c2.metric("Best model", models["best_model_name"])
    c3.metric("R² (test)", f"{best['R² Test']:.3f}")
    c4.metric("MAE (test)", f"{best['MAE Test']:.2f} MPa")
    st.warning("About half of the challenge rows are physically implausible (components add up to roughly twice a real mix). "
               "They were kept for training, as the challenge requires.")
    st.dataframe(data["plausibility"].round(2))

elif page == "Data":
    st.title("Data exploration")
    df = data["df"]
    st.plotly_chart(px.histogram(df, x=TARGET, nbins=30, color="Strength Category",
                                 labels={TARGET: "Strength (MPa)"}), width="stretch")
    corr = df[data["feature_columns"] + [TARGET]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation matrix"), width="stretch")
    st.dataframe(df.head(20))

elif page == "Models":
    st.title("Model comparison")
    st.write(f"Train: {len(data['X_train']):,} rows · Test: {len(data['X_test']):,} rows · split 80/20, random_state=42")
    table = results[["Model", "Type", "R² Test", "RMSE Test", "MAE Test", "Overfitting (R² gap)"]]
    st.dataframe(table.round(4), hide_index=True)
    st.plotly_chart(px.bar(results, x="R² Test", y="Model", color="Type", orientation="h",
                           title="R² on the test set"), width="stretch")

elif page == "Feature importance":
    st.title("Feature importance (tree-based models)")
    importance = data["feature_importance"].reset_index(names="Feature")
    long = importance.melt(id_vars="Feature", value_vars=["Gradient Boosting", "Random Forest", "Decision Tree"],
                           var_name="Model", value_name="Importance")
    st.plotly_chart(px.bar(long, x="Importance", y="Feature", color="Model", barmode="group", orientation="h"),
                    width="stretch")

else:
    st.title("Strength simulator")
    st.write(f"Prediction from the {models['best_model_name']} model. The band shows ± the test MAE ({best['MAE Test']:.2f} MPa).")
    c1, c2 = st.columns(2)
    with c1:
        cement = st.slider("Cement (kg/m³)", 100, 600, 350)
        slag = st.slider("Blast furnace slag (kg/m³)", 0, 300, 100)
        fly_ash = st.slider("Fly ash (kg/m³)", 0, 200, 30)
        water = st.slider("Water (kg/m³)", 120, 250, 180)
    with c2:
        superplasticizer = st.slider("Superplasticizer (kg/m³)", 0, 30, 8)
        coarse = st.slider("Coarse aggregate (kg/m³)", 800, 1200, 950)
        fine = st.slider("Fine aggregate (kg/m³)", 600, 1000, 750)
        age = st.slider("Age (days)", 1, 365, 28)
    row = pd.DataFrame([[cement, slag, fly_ash, water, superplasticizer, coarse, fine, age]],
                       columns=data["feature_columns"])
    prediction = models["best_model"].predict(row)[0]
    st.metric("Predicted strength", f"{prediction:.1f} MPa")
    st.info(f"Expected range: {prediction - best['MAE Test']:.1f} to {prediction + best['MAE Test']:.1f} MPa · "
            f"water/cement ratio: {water / cement:.2f}")
