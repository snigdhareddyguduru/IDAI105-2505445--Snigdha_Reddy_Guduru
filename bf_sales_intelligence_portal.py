import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats

st.set_page_config(page_title="Black Friday Sales Intelligence Portal", layout="wide")

st.title("Black Friday Consumer Intelligence Portal")

# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("blackfriday_dataset.csv")
    return df

df = load_data()

# -----------------------------
# DATA CLEANING
# -----------------------------

df['Product_Category_2'].fillna(0, inplace=True)
df['Product_Category_3'].fillna(0, inplace=True)

df['Gender'] = df['Gender'].map({'M':0,'F':1})
df['Age'] = df['Age'].astype('category').cat.codes
df['City_Category'] = df['City_Category'].astype('category').cat.codes

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Project Overview",
        "Dataset Explorer",
        "Exploratory Data Analysis",
        "Customer Segmentation",
        "Product Relationship Analysis",
        "Anomaly Detection",
        "Business Insights"
    ]
)

# -----------------------------
# PAGE 1 OVERVIEW
# -----------------------------

if page == "Project Overview":

    st.header("Project Objective")

    st.write("""
This project analyzes customer purchasing behaviour during the Black Friday sales event.
The goal is to identify patterns in customer spending, understand demographic influences,
segment customers into behavioral groups, and detect unusual purchasing patterns using
data mining techniques.
""")

    st.subheader("Techniques Used")

    st.write("""
- Exploratory Data Analysis (EDA)
- Customer Segmentation using K-Means Clustering
- Product Relationship Analysis
- Anomaly Detection
- Interactive Visualization through Streamlit
""")

# -----------------------------
# PAGE 2 DATASET
# -----------------------------

elif page == "Dataset Explorer":

    st.header("Dataset Overview")

    st.write("Dataset Shape:", df.shape)

    st.subheader("Dataset Preview")

    rows = st.slider("Select number of rows to display", 5, 50, 10)
    st.dataframe(df.head(rows))

    st.subheader("Column Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.write(df.describe())

# -----------------------------
# PAGE 3 EDA
# -----------------------------

elif page == "Exploratory Data Analysis":

    st.header("Consumer Behaviour Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Purchase Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Purchase"], bins=40, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Purchase by Gender")
        fig, ax = plt.subplots()
        sns.barplot(x="Gender", y="Purchase", data=df, ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Purchase by Age Group")
        fig, ax = plt.subplots()
        sns.boxplot(x="Age", y="Purchase", data=df, ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Product Category Popularity")
        fig, ax = plt.subplots()
        sns.countplot(x="Product_Category_1", data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    st.write("This heatmap shows relationships between numeric variables in the dataset.")

    numeric_df = df.select_dtypes(include=['number'])

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)

    st.pyplot(fig)

# -----------------------------
# PAGE 4 CLUSTERING
# -----------------------------

elif page == "Customer Segmentation":

    st.header("Customer Segmentation Using K-Means")

    features = df[['Age','Occupation','Purchase','Marital_Status']]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    st.subheader("Elbow Method to Determine Optimal Clusters")

    inertia = []
    K_range = range(1,10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    st.subheader("Customer Segments")

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    df['Cluster'] = clusters

    st.write(df['Cluster'].value_counts())

    fig, ax = plt.subplots()
    scatter = ax.scatter(df["Age"], df["Purchase"], c=df["Cluster"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Purchase")
    st.pyplot(fig)

# -----------------------------
# PAGE 5 PRODUCT RELATIONSHIPS
# -----------------------------

elif page == "Product Relationship Analysis":

    st.header("Product Category Relationships")

    product_matrix = pd.crosstab(df["Product_Category_1"], df["Product_Category_2"])

    st.subheader("Product Category Co-occurrence Matrix")
    st.dataframe(product_matrix.head(10))

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(product_matrix, cmap="viridis")
    st.pyplot(fig)

    st.write("""
This analysis highlights product categories frequently purchased together,
helping retailers design cross-selling strategies and bundled offers.
""")

# -----------------------------
# PAGE 6 ANOMALY DETECTION
# -----------------------------

elif page == "Anomaly Detection":

    st.header("High-Spending Transaction Detection")

    z_scores = np.abs(stats.zscore(df["Purchase"]))
    anomalies = df[z_scores > 3]

    st.subheader("Number of Anomalies Detected")
    st.write(len(anomalies))

    st.subheader("Sample Anomalies")
    st.dataframe(anomalies.head(20))

    fig, ax = plt.subplots()
    ax.scatter(df.index, df["Purchase"])
    ax.scatter(anomalies.index, anomalies["Purchase"], color='red')
    st.pyplot(fig)

# -----------------------------
# PAGE 7 INSIGHTS
# -----------------------------

elif page == "Business Insights":

    st.header("Key Findings")

    st.write("""
1. Certain demographic groups demonstrate higher purchasing levels.
2. Product Category 1 represents the most frequently purchased category.
3. Customer segmentation reveals different spending profiles across age groups.
4. Some product categories frequently co-occur in transactions, suggesting cross-selling opportunities.
5. A small set of transactions exhibit unusually high purchase values, indicating premium customers or bulk buyers.
""")
