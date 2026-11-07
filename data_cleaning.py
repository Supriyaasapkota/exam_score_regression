
import pandas as pd


df = pd.read_csv("Exam_Score_Prediction.csv")

print("Original Dataset Shape:", df.shape)


df = df.drop(columns=["student_id"])


print("\nMissing Values:")
print(df.isnull().sum())


print("\nDuplicate Rows:", df.duplicated().sum())


df.to_csv("exam_score_cleaned.csv", index=False)

print("\nCleaned Dataset Shape:", df.shape)
print("\nCleaned Dataset Saved Successfully!")
