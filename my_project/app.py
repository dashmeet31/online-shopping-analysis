import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io

# Config
st.set_page_config(page_title="Online Shopping Data Analysis", layout="wide")
sns.set_theme(style="whitegrid")

st.title("🛒 Online Shopping Data Analysis Dashboard")

# ----------------------------- Load Data -----------------------------
@st.cache_data
def load_and_clean_data():
    file_path = os.path.join(os.path.dirname(__file__), "online_shopping_4000_instances.csv")
    df = pd.read_csv(file_path)

    # Ensure Order Date is datetime
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # Ensure Total Amount is numeric
    if "Total Amount" in df.columns:
        df["Total Amount"] = pd.to_numeric(df["Total Amount"], errors="coerce")

    # Drop rows only if columns exist
    cols_to_check = [c for c in ["Total Amount", "City"] if c in df.columns]
    if cols_to_check:
        df = df.dropna(subset=cols_to_check)

    return df

# Load cleaned dataset
df = load_and_clean_data()

# ----------------------------- Section 1: Dataset Viewer -----------------------------
st.header("1. View & Download Cleaned Dataset")

with st.expander("📂 Preview Dataset"):
    st.dataframe(df, use_container_width=True)

csv_download = df.to_csv(index=False).encode('utf-8')
st.download_button(
    "⬇️ Download Cleaned CSV",
    data=csv_download,
    file_name="cleaned_online_shopping_data.csv",
    mime='text/csv'
)

# ----------------------------- Section 2: Statistics -----------------------------
if st.button("📊 Show Dataset Statistics"):
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# ----------------------------- Section 3: Visualizations -----------------------------
if st.button("📈 Show Visualizations"):

    if "Product Category" in df.columns:
        st.subheader("Popular Product Categories")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        category_order = df['Product Category'].value_counts().index
        sns.countplot(data=df, y='Product Category', order=category_order, palette='Blues_d', ax=ax1)
        st.pyplot(fig1)

    if {"Total Amount", "City"}.issubset(df.columns):
        st.subheader("Distribution of Sales Amounts by City")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='Total Amount', y='City', palette='mako', ax=ax2)
        st.pyplot(fig2)

    if "Gender" in df.columns:
        st.subheader("Gender Distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Gender', data=df, palette='Set2', ax=ax3)
        st.pyplot(fig3)

    if "Age" in df.columns:
        st.subheader("Age Distribution")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Age'], bins=10, kde=True, color='green', ax=ax4)
        st.pyplot(fig4)

    if "Payment mode" in df.columns:
        st.subheader("Payment Modes")
        fig5, ax5 = plt.subplots(figsize=(6, 6))
        df['Payment mode'].value_counts().plot(
            kind='pie', autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette("pastel"), ax=ax5
        )
        ax5.set_ylabel("")
        st.pyplot(fig5)

    if {"Order Date", "Total Amount"}.issubset(df.columns):
        st.subheader("Monthly Sales Trend")
        df['Month'] = df['Order Date'].dt.to_period('M')
        monthly_sales = df.groupby('Month')['Total Amount'].sum().reset_index()
        monthly_sales['Month'] = monthly_sales['Month'].astype(str)

        fig6, ax6 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=monthly_sales, x='Month', y='Total Amount', marker='o', color='purple', ax=ax6)
        plt.xticks(rotation=45)
        st.pyplot(fig6)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax7)
        st.pyplot(fig7)

# ----------------------------- Section 4: Summary Insights -----------------------------
if st.button("📝 Show Summary Insights"):
    st.subheader("Project Summary")

    if "Product Category" in df.columns:
        top_category = df['Product Category'].value_counts().idxmax()
    else:
        top_category = "N/A"

    if {"City", "Total Amount"}.issubset(df.columns):
        top_city = df.groupby('City')['Total Amount'].sum().idxmax()
    else:
        top_city = "N/A"

    if "Payment mode" in df.columns:
        top_payment = df['Payment mode'].value_counts().idxmax()
    else:
        top_payment = "N/A"

    gender_dist = df['Gender'].value_counts(normalize=True) * 100 if "Gender" in df.columns else {}
    avg_age = df['Age'].mean() if "Age" in df.columns else "N/A"

    st.markdown(f"""
    - **Most purchased category:** `{top_category}`  
    - **City with highest sales:** `{top_city}`  
    - **Most used payment mode:** `{top_payment}`  
    - **Gender distribution:**  
        - Male: `{gender_dist.get('Male', 0):.1f}%`  
        - Female: `{gender_dist.get('Female', 0):.1f}%`  
    - **Average age of customers:** `{avg_age}`
    """)

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.caption("Developed by Dashmeet Singh | Data Analyst Internship Project")
