import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

file_path = 'data\marketing_campaign.csv'
df = pd.read_csv(file_path, delimiter='\t')

# Initial Data Overview
print("Initial Data Overview:")
print(df.head())
print(df.shape)
print(df.info())
print(df.columns)

# Handling Missing Values
# Assuming 'Income' might have missing values, fill them with the median
df['Income'] = df['Income'].fillna(df['Income'].median())

# # Convert 'Dt_Customer' to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# Feature Engineering
# Create a new feature for customer age and tenure

current_year = pd.Timestamp.now().year
df['Age'] = current_year - df['Year_Birth']
df['Customer_Tenure_Days'] = (pd.Timestamp.now() - df['Dt_Customer']).dt.days

# Further EDA: Visualizations
# Distribution of customer age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Income distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Income'], bins=30, kde=True)
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Average spending by education level
df.groupby('Education')[['MntWines', 'MntMeatProducts', 'MntGoldProds']].mean().plot(kind='bar')
plt.title('Average Spending by Education Level')
plt.ylabel('Average Amount Spent')
plt.show()

# Campaign success rate by marital status
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
df['TotalAcceptedCmps'] = df[campaign_cols].sum(axis=1)
df.groupby('Marital_Status')['TotalAcceptedCmps'].mean().plot(kind='bar')
plt.title('Average Campaign Success by Marital Status')
plt.ylabel('Average Campaigns Accepted')
plt.show()

# Age groups and spending on wine
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70, 120], labels=['18-30', '31-40', '41-50', '51-60', '61-70', '70+'])
df.groupby('AgeGroup')['MntWines'].mean().plot(kind='bar')
plt.title('Wine Spending by Age Group')
plt.ylabel('Average Amount Spent on Wine')
plt.show()

#Spending and Deomgraphics comparsion
sns.pairplot(df[['MntWines', 'MntFruits', 'MntMeatProducts', 'Income', 'Age']], corner=True)
plt.suptitle('Pairwise Relationships Between Spending and Demographics')
plt.show()

# Preprocessing for Modeling
# Selecting features for customer segmentation - focusing on income, age, and spending on wines as an example

features = ['Income', 'Age', 'MntWines']

# Scaling features

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Convert scaled data back to a DataFrame (for clustering or further analysis)
df_scaled = pd.DataFrame(df_scaled, columns=features)

print("Scaled Features Head:")
print(df_scaled.head())


