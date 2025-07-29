#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Config
st.set_page_config(page_title="Online Shopping Data Analysis", layout="wide")
sns.set_theme(style="whitegrid")

st.title("🛍️ Online Shopping Data Analysis Dashboard")

# Load Data and Clean
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("online_shopping_4000_instances.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Total Amount'] = df['Amount spent by per user'] - df['Discount']
    df.to_csv("cleaned_online_shopping_data.csv", index=False)
    return df

# Load cleaned dataset
df = load_and_clean_data()

# ----------------------------- Section 1: Dataset Viewer -----------------------------
st.header("📁 1. View & Download Cleaned Dataset")

with st.expander("🔍 Preview Dataset"):
    st.dataframe(df, use_container_width=True)

csv_download = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Cleaned CSV", data=csv_download, file_name="cleaned_online_shopping_data.csv", mime='text/csv')

# ----------------------------- Section 2: Statistics -----------------------------
if st.button("📊 Show Dataset Statistics"):
    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📌 Summary Statistics")
    st.write(df.describe())

    st.subheader("📌 Dataset Info")
    buffer = []
    df.info(buf := buffer)
    st.code("\n".join(buffer), language='text')

# ----------------------------- Section 3: Visualizations -----------------------------
if st.button("📈 Show Visualizations"):
    st.subheader("🎯 Popular Product Categories")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    category_order = df['Product Category'].value_counts().index
    sns.countplot(data=df, y='Product Category', order=category_order, palette='Blues_d', ax=ax1)
    st.pyplot(fig1)

    st.subheader("🏙️ Distribution of Sales Amounts by City")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Total Amount', y='City', palette='mako', ax=ax2)
    st.pyplot(fig2)

    st.subheader("👥 Gender Distribution")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.countplot(x='Gender', data=df, palette='Set2', ax=ax3)
    st.pyplot(fig3)

    st.subheader("🎂 Age Distribution")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['Age'], bins=10, kde=True, color='green', ax=ax4)
    st.pyplot(fig4)

    st.subheader("💳 Payment Modes")
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    df['Payment mode'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90,
                                           colors=sns.color_palette("pastel"), ax=ax5)
    ax5.set_ylabel("")
    st.pyplot(fig5)

    st.subheader("📆 Monthly Sales Trend")
    df['Month'] = df['Order Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Total Amount'].sum().reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].astype(str)

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=monthly_sales, x='Month', y='Total Amount', marker='o', color='purple', ax=ax6)
    plt.xticks(rotation=45)
    st.pyplot(fig6)

    st.subheader("📊 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.corr()

    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax7)
    st.pyplot(fig7)

# ----------------------------- Section 4: Summary Insights -----------------------------
if st.button("📌 Show Summary Insights"):
    st.subheader("📋 Project Summary")

    top_category = df['Product Category'].value_counts().idxmax()
    top_city = df.groupby('City')['Total Amount'].sum().idxmax()
    top_payment = df['Payment mode'].value_counts().idxmax()
    gender_dist = df['Gender'].value_counts(normalize=True) * 100
    avg_age = df['Age'].mean()

    st.markdown(f"""
    - 🛍️ **Most purchased category:** `{top_category}`  
    - 🏙️ **City with highest sales:** `{top_city}`  
    - 💳 **Most used payment mode:** `{top_payment}`  
    - 👨‍👩‍👧‍👦 **Gender distribution:**  
        - Male: `{gender_dist['Male']:.1f}%`  
        - Female: `{gender_dist['Female']:.1f}%`  
    - 🎂 **Average age of customers:** `{avg_age:.1f} years`
    """)

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.caption("Developed by Dashmeet Singh 💼 | Data Analyst Internship Project")

