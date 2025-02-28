import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "heart_attack_dataset.csv"
df = pd.read_csv(file_path)

# Univariate Analysis
## Summary statistics for numerical variables
print("\nSummary Statistics:\n", df.describe())

## Histograms for numerical variables
plt.figure(figsize=(20, 15))
df.hist(figsize=(20, 15), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

## Count plots for categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette="coolwarm")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

# Multivariate Analysis
## Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

## Pairplot for selected variables to check relationships
selected_features = ['Age', 'Cholesterol', 'BloodPressure', 'BMI', 'HeartRate', 'Outcome']
sns.pairplot(df[selected_features], hue='Outcome', palette='husl')
plt.show()

## Boxplot to analyze distributions by outcome
plt.figure(figsize=(10, 5))
sns.boxplot(x='Outcome', y='Cholesterol', data=df, palette='coolwarm')
plt.title("Cholesterol Levels by Outcome")
plt.tight_layout()
plt.show()

## Countplot for Gender vs Outcome
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', hue='Outcome', data=df, palette='coolwarm')
plt.title("Heart Attack Occurrence by Gender")
plt.tight_layout()
plt.show()
