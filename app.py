import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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

    # 🕵️ Auto-Detect Employee Identifier
    search_column = None
    for col in df.columns:
        if "id" in col.lower() or "name" in col.lower():
            search_column = col
            break

    # 🎯 Employee Search & Prediction
    if search_column:
        search_value = st.text_input(f"🔍 Search Employee by {search_column}")

        if search_value:
            try:
                if df[search_column].dtype != object:
                    search_value = int(search_value)

                employee_data = df[df[search_column] == search_value]

                if not employee_data.empty:
                    # 📌 Show Employee Details
                    st.write("✅ Employee Found:")
                    st.dataframe(employee_data)

                    # 🔥 Predict for Selected Employee
                    feature_cols = [col for col in df.columns if col.lower() not in ["employee_id", "name", "absenteeism_risk"]]
                    employee_features = employee_data[feature_cols]

                    if st.button("📊 Predict Absenteeism"):
                        prediction = model.predict(employee_features)[0]
                        risk_prob = model.predict_proba(employee_features)[0][1] * 100

                        if prediction == 1:
                            st.error(f"⚠️ High Absenteeism Risk! ({risk_prob:.2f}% probability)")
                        else:
                            st.success(f"✅ Low Absenteeism Risk! ({risk_prob:.2f}% probability)")

                        # 🌟 Update Session with Selected Employee
                        st.session_state.selected_employee = employee_data

                        # 📊 Show Employee Details in Charts
                        st.subheader("📊 Employee Insights")
                        fig = px.bar(
                            employee_data.melt(id_vars=[search_column], value_vars=feature_cols),
                            x="variable", y="value",
                            title="Employee Features Overview",
                            color="variable"
                        )
                        st.plotly_chart(fig)

                else:
                    st.warning("❌ Employee Not Found. Please check the ID or Name.")
            except ValueError:
                st.warning("⚠️ Please enter a valid Employee ID or Name.")

    # 📈 **Absenteeism Trends**
    st.subheader("📊 Absenteeism Trends")

    # 📉 Absenteeism Distribution
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    if "Absenteeism_Days" in df.columns:
        sns.histplot(df["Absenteeism_Days","Past Absences"], bins=20, kde=True, color="skyblue", ax=ax1)
    else:
        st.warning("⚠️ The column 'Absenteeism_Days' is missing in your dataset. Please upload a valid file.")
    ax1.set_title("Company Absenteeism Distribution")
    st.pyplot(fig1)

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
