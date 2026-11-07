
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("Exam_Score_Prediction.csv")

print("First 5 Rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()


print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


print("\nDuplicate Rows:")
print(df.duplicated().sum())


print("\nColumn Names:")
print(df.columns.tolist())


categorical_columns = df.select_dtypes(
    include="str"
).columns

print("\nUnique Categorical Values:")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())



sns.set_style("whitegrid")


plt.figure(figsize=(8, 5))
sns.histplot(df["exam_score"], kde=True)
plt.title("Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=df["exam_score"])
plt.title("Exam Score Box Plot")
plt.xlabel("Exam Score")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x="study_hours", y="exam_score", data=df)
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x="class_attendance", y="exam_score", data=df)
plt.title("Class Attendance vs Exam Score")
plt.xlabel("Class Attendance (%)")
plt.ylabel("Exam Score")
plt.show()


numeric_df = df.select_dtypes(
    include="number"
).drop(columns=["student_id"])

plt.figure(figsize=(10, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()
