

Black Friday Customer Intelligence  Analysis

<img width="1440" height="900" alt="Screenshot 2026-03-16 at 10 50 02 PM" src="https://github.com/user-attachments/assets/cb17edbb-c620-42dd-b66a-2e1e580141a0" />

<img width="1440" height="900" alt="Screenshot 2026-03-16 at 10 49 16 PM" src="https://github.com/user-attachments/assets/dc5e1e64-e5a3-4aca-ba93-2b913d5dbeca" />

<img width="1440" height="900" alt="Screenshot 2026-03-16 at 10 49 39 PM" src="https://github.com/user-attachments/assets/9963d570-e8bd-4313-a4de-ea8f739fdbef" />

This project investigates the purchasing behavior of customers during the Black Friday sales event through data mining. The objective of the project is to analyze the purchasing behavior of customers during massive sales events and to discover patterns that can enable businesses to make informed decisions. The project aims to discover insights about customer spending behavior, product demand, and customer segmentation by analyzing demographic information such as age, gender, and occupation, as well as purchase information and product categories. The final output of the project is presented in the form of an interactive dashboard using the Streamlit library.

The dataset for this project holds transaction-level data regarding customer purchases made during the Black Friday sales. It has variables like User ID, Product ID, Gender, Age group, Occupation, City Category, marital status, product category details, and the total purchase amount. These variables enable us to analyze both customer details and purchase behavior to look for patterns in the data. Before carrying out the analysis, the dataset was preprocessed by addressing missing values in the product category variables and transforming non-numeric variables like gender, age group, and city category into numeric values that could be processed by machine learning algorithms.

Exploratory Data Analysis (EDA) was also performed to gain insights into the global characteristics of the data and to discover early patterns. Various data visualizations were performed, such as histograms of purchase distribution, comparison of spending amounts by gender and age categories, popularity charts of product categories, and correlation heatmaps. These data visualizations were used to emphasize the relationships between the variables and to gain a better understanding of the impact of various demographic factors on retail sales. The correlation heatmap was particularly useful in determining the relationships between the numerical variables and choosing features for further analysis.

Customer segmentation was carried out using the K-Means clustering algorithm. The aim of clustering was to segment customers based on their purchasing behavior. Variables like age, occupation, marital status, and the amount spent were considered to segment customers. The Elbow Method was used to identify the number of clusters for the algorithm. The result of clustering shows the different types of customers in the dataset, including moderate customers, high-value customers, and occasional customers. This information can be used by businesses to create marketing campaigns to target the customers.

The project also involves an analysis of the relationship between product categories in the dataset. A co-occurrence matrix was used to display the relationship between product categories. The analysis of the relationship between product categories can be used by retailers to create strategies for recommending products to customers based on the categories that appear together in the dataset.

Additionally, an anomaly detection task was also considered in the analysis to point out the unusual high value of purchases. By applying the statistical z-score analysis, the data with highly unusual high spending amounts was identified and pointed out in the data. These points in the data may highlight the high-value customers, bulk purchase customers, or the unusual purchase behavior of customers that may be beneficial for the retailers to investigate further.

To make the analysis more interactive and easier to understand, the entire project was applied to a Streamlit dashboard. The application allows the users to explore the different sections of the analysis such as project overview, data exploration, exploratory data analysis, customer segmentation, product relationship analysis, anomaly detection, and key business insights. The application provides an interactive platform where the results can be explored dynamically.

Conclusion
------------

This project illustrates how data mining algorithms can be used on retail data to produce valuable information. The project combines data preprocessing, data visualization, clustering analysis, and anomaly detection to offer a complete view of customer purchasing behavior during large sales events. This information can be used by retailers to enhance marketing campaigns, product development, and understanding of their target market.

Repository Structure
---------------------

The repository holds the data, the Streamlit application code, and the requirements file required to execute the project.

```
blackfriday_data_mining_ai
│
├── blackfriday_dataset.csv
├── bf_sales_intelligence_portal.py
├── requirements_blackfriday_ai.txt
```

 Technologies Used

The programming language used for this project was Python. The analysis was done using libraries such as Pandas and NumPy for data analysis, Matplotlib and Seaborn for data visualization, Scikit-learn for cluster analysis, SciPy for anomaly detection, and Streamlit for creating the interactive dashboard.

Conclusion

The above analysis shows that data mining can be used to gain insights into the purchasing behavior of customers during retail sales events. By analyzing customer segments, product relationships, and anomalies in purchasing behavior, businesses can make informed decisions. The interactive dashboard created in this project can be used to present the insights gained from the analysis.
