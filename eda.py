import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load Dataset

df = pd.read_csv("dataset.csv", sep="\t")


# Dataset Overview

print("="*50)
print("FIRST 5 ROWS")
print("="*50)
print(df.head())

print("\nSHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATA INFORMATION")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())


# Data Cleaning

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())


# Histogram

plt.figure(figsize=(8,5))
plt.hist(df["G3"], bins=10, edgecolor="black")
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.grid(True)
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df["studytime"], df["G3"])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.grid(True)
plt.show()


# Box Plot

plt.figure(figsize=(8,5))
sns.boxplot(y=df["G3"])
plt.title("Box Plot of Final Grades")
plt.show()


# Bar Chart

plt.figure(figsize=(8,5))
df["studytime"].value_counts().sort_index().plot(kind="bar")
plt.title("Students by Study Time")
plt.xlabel("Study Time")
plt.ylabel("Number of Students")
plt.show()


# Pie Chart

plt.figure(figsize=(6,6))
df["sex"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()


# Count Plot

plt.figure(figsize=(8,5))
sns.countplot(x="studytime", data=df)
plt.title("Count of Study Time")
plt.show()


# Line Plot

plt.figure(figsize=(8,5))
plt.plot(df["G3"].sort_values().values)
plt.title("Final Grades Trend")
plt.xlabel("Students")
plt.ylabel("Final Grade")
plt.grid(True)
plt.show()

# Average Grades

print("\nAverage Grades")
print("G1 Mean :", df["G1"].mean())
print("G2 Mean :", df["G2"].mean())
print("G3 Mean :", df["G3"].mean())
# Correlation with Final Grade

print("\nCorrelation with G3")
print(numeric_df.corr()["G3"].sort_values(ascending=False))