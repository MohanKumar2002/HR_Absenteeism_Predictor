import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# 🎯 Load AI Model
model = joblib.load("absenteeism_model.pkl")

# 📂 Secure Storage
DATA_DIR = "datasets"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 🛡️ Initialize Session Data
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.selected_employee = None

# 🎨 Streamlit UI Config
st.set_page_config(page_title="AI Absenteeism Predictor", layout="wide")

# 🏢 Header Section
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>📊 AI Absenteeism Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #7F8C8D;'>Upload employee data & predict absenteeism risks instantly.</h5>", unsafe_allow_html=True)
st.markdown("---")

# 📂 Upload Dataset
uploaded_file = st.file_uploader("📁 Upload Employee CSV Data", type=["csv"], help="Upload your dataset for absenteeism analysis.")

if uploaded_file:
    # 🛡️ Secure Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    client_filename = f"Client_{timestamp}.csv"
    file_path = os.path.join(DATA_DIR, client_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 🚀 Load Dataset for This Session
    st.session_state.df = pd.read_csv(uploaded_file)

    # ✅ Success Message
    st.success("✅ File uploaded successfully!")

# 🛠️ UI Refresh Logic
if st.session_state.df is None:
    st.warning("⚠️ No dataset uploaded yet. Please upload a CSV file.")
else:
    df = st.session_state.df

    # 📌 Show First 10 Rows in Stylish Table
    st.subheader("📌 Preview of Uploaded Data")
    st.dataframe(df.head(10))

    # 🧑‍💼 Cards and KPI Section
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_employees = len(df)
        st.metric("Total Employees", total_employees)

    with col2:
        high_risk_employees = len(df[df["Absenteeism Risk Score"] > 75])
        st.metric("High-Risk Employees", high_risk_employees)

    with col3:
        avg_absenteeism_days = df["Absenteeism_Days"].mean()
        st.metric("Avg. Absenteeism Days", round(avg_absenteeism_days, 2))

    with col4:
        dept_absenteeism = df.groupby("Department")["Absenteeism_Days"].sum().idxmax()
        st.metric("Highest Absenteeism Department", dept_absenteeism)

    # 🏢 KPI for Absenteeism Risk Score
    avg_risk_score = df["Absenteeism Risk Score"].mean()
    st.subheader("📊 KPI: Average Absenteeism Risk Score")
    st.write(f"Avg. Absenteeism Risk Score for all employees: {avg_risk_score:.2f}%")

    # 🔄 Gauge Chart for Absenteeism Risk
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_risk_score,
        title={"text": "Absenteeism Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 75], "color": "yellow"},
                {"range": [75, 100], "color": "red"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge)

    # 📈 Absenteeism Distribution
    st.subheader("📊 Absenteeism Trends")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Absenteeism_Days"], bins=20, kde=True, color="skyblue", ax=ax1)
    ax1.set_title("Company Absenteeism Distribution")
    st.pyplot(fig1)

    # 📊 Absenteeism by Department
    st.subheader("📊 Absenteeism by Department")
    dept_data = df.groupby("Department")["Absenteeism_Days"].sum().reset_index()
    fig2 = px.bar(dept_data, x="Department", y="Absenteeism_Days", title="Absenteeism by Department")
    st.plotly_chart(fig2)

    # 📊 Performance vs Absenteeism Correlation
    st.subheader("📊 Performance vs Absenteeism Correlation")
    fig3 = px.scatter(df, x="Performance_Rating", y="Absenteeism_Days", color="Department", title="Performance vs Absenteeism")
    st.plotly_chart(fig3)

    # 🏢 Bulk Prediction Option
    if st.checkbox("📂 Predict for All Employees"):
        feature_cols = [col for col in df.columns if col.lower() not in ["employee_id", "name", "absenteeism_risk"]]
        df["Prediction"] = model.predict(df[feature_cols])
        df["Risk Probability (%)"] = model.predict_proba(df[feature_cols])[:, 1] * 100

        # 🔽 Download Predictions
        st.subheader("📊 Predictions for All Employees")
        st.dataframe(df[[search_column, "Prediction", "Risk Probability (%)"]] if search_column else df)
        st.download_button("📥 Download Predictions", df.to_csv(index=False), file_name="Predictions.csv")

# 🔄 Reset Button (Removes Data for Security)
if st.button("🔄 Reset"):
    st.session_state.df = None
    st.session_state.selected_employee = None
    st.experimental_rerun()
