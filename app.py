
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan EDA Dashboard", layout="wide")
st.title("🏦 Loan Dataset – EDA Dashboard (Cleaned Data)")

# ---------------- Sidebar Upload ----------------
st.sidebar.header("Upload Cleaned Dataset")
file = st.sidebar.file_uploader("Upload cleaned CSV file", type=["csv"])

if file is None:
    st.info("👈 Upload your cleaned dataset to start analysis.")
    st.stop()

df = pd.read_csv(file)

# ---------------- Dataset Preview ----------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# ---------------- Dataset Shape ----------------
st.subheader("📐 Dataset Shape")
c1, c2 = st.columns(2)
with c1:
    st.metric("Rows", df.shape[0])
with c2:
    st.metric("Columns", df.shape[1])

# ---------------- Data Types ----------------
st.subheader("🧾 Column Data Types")
st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

# ---------------- Summary Statistics ----------------
st.subheader("📊 Summary Statistics")
st.dataframe(df.describe(include="all").transpose())

# ---------------- Missing Values ----------------
st.subheader("🩹 Missing Values")
st.dataframe(df.isnull().sum())

# ---------------- Column Selection ----------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# ---------------- Correlation Heatmap ----------------
if len(num_cols) > 1:
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- Numerical Distribution ----------------
st.subheader("📈 Numerical Feature Distribution")
selected_num = st.selectbox("Select Numerical Column", num_cols)

fig2, ax2 = plt.subplots()
sns.histplot(df[selected_num], kde=True, ax=ax2)
st.pyplot(fig2)

# ---------------- Boxplot ----------------
st.subheader("📦 Outlier Detection (Boxplot)")
fig3, ax3 = plt.subplots()
sns.boxplot(x=df[selected_num], ax=ax3)
st.pyplot(fig3)

# ---------------- Categorical Countplot ----------------
st.subheader("🧮 Categorical Feature Distribution")
selected_cat = st.selectbox("Select Categorical Column", cat_cols)

fig4, ax4 = plt.subplots()
sns.countplot(data=df, x=selected_cat, ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)

# ---------------- Target Analysis (Optional) ----------------
if "Loan_Status" in df.columns:
    st.subheader("🎯 Loan Status Analysis")

    fig5, ax5 = plt.subplots()
    sns.countplot(data=df, x="Loan_Status", ax=ax5)
    st.pyplot(fig5)

    if "Credit_History" in df.columns:
        st.subheader("📌 Loan Status vs Credit History")
        fig6, ax6 = plt.subplots()
        sns.countplot(data=df, x="Credit_History", hue="Loan_Status", ax=ax6)
        st.pyplot(fig6)

# ---------------- Missing Heatmap ----------------
st.subheader("🔎 Missing Values Heatmap")
fig7, ax7 = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax7)
st.pyplot(fig7)

st.success("✅ Cleaned dataset EDA ready. Explore all statistics & visualizations!")
