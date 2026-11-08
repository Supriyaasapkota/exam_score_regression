
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


st.set_page_config(
    page_title="Exam Score Prediction",
    layout="wide"
)



raw_df = pd.read_csv("Exam_Score_Prediction.csv")
df = pd.read_csv("exam_score_cleaned.csv")


model = joblib.load("best_model.joblib")



comparison = pd.read_csv("model_comparison.csv")


st.sidebar.title("Exam Score Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Details",
        "Dataset",
        "EDA",
        "Model Comparison",
        "Prediction"
    ]
)



if page == "Project Details":

    st.title("Exam Score Prediction")

    st.write(
        "This project uses machine learning techniques "
        "to predict students' examination scores based "
        "on their academic habits and other factors."
    )

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1] - 1)
    col3.metric("Target Variable", "Exam Score")



elif page == "Dataset":

    st.title("Dataset Overview")

    st.subheader("First 5 Records")
    st.dataframe(raw_df.head())

    st.subheader("Dataset Shape")

    st.write(f"Rows: {raw_df.shape[0]}")
    st.write(f"Columns: {raw_df.shape[1]}")

    st.subheader("Data Types")

    st.dataframe(
        raw_df.dtypes.astype(str).reset_index().rename(
            columns={"index": "Column", 0: "Data Type"}
        )
    )

    st.subheader("Missing Values")

    st.dataframe(
        raw_df.isnull().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing Values"}
        )
    )

    st.subheader("Duplicate Rows")
    st.write(raw_df.duplicated().sum())

    st.subheader("Statistical Summary")
    st.dataframe(raw_df.describe())



elif page == "EDA":

    st.title("Exploratory Data Analysis")

    st.write(
        "Explore the raw dataset using interactive filters "
        "and visualizations."
    )

    st.sidebar.subheader("Dataset Filters")

    selected_course = st.sidebar.multiselect(
        "Select Course",
        options=sorted(raw_df["course"].unique()),
        default=sorted(raw_df["course"].unique())
    )

    selected_gender = st.sidebar.multiselect(
        "Select Gender",
        options=sorted(raw_df["gender"].unique()),
        default=sorted(raw_df["gender"].unique())
    )

    selected_difficulty = st.sidebar.multiselect(
        "Select Exam Difficulty",
        options=sorted(raw_df["exam_difficulty"].unique()),
        default=sorted(raw_df["exam_difficulty"].unique())
    )

    filtered_df = raw_df[
        raw_df["course"].isin(selected_course)
        & raw_df["gender"].isin(selected_gender)
        & raw_df["exam_difficulty"].isin(selected_difficulty)
    ]

    if filtered_df.empty:

        st.warning(
            "No records found. Please select at least one "
            "option from each filter."
        )

    else:

        st.subheader("Filtered Dataset Overview")

        col1, col2 = st.columns(2)

        col1.metric("Filtered Records", len(filtered_df))

        col2.metric(
            "Average Exam Score",
            round(filtered_df["exam_score"].mean(), 2)
        )

        sns.set_style("whitegrid")

        
        st.subheader("Exam Score Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.histplot(
            filtered_df["exam_score"],
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Exam Score")
        ax.set_ylabel("Number of Students")

        st.pyplot(fig)
        plt.close(fig)

       
        st.subheader("Exam Score Box Plot")

        fig, ax = plt.subplots(figsize=(8, 3))

        sns.boxplot(
            x=filtered_df["exam_score"],
            ax=ax
        )

        ax.set_xlabel("Exam Score")

        st.pyplot(fig)
        plt.close(fig)

        
        st.subheader("Study Hours vs Exam Score")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.scatterplot(
            x="study_hours",
            y="exam_score",
            data=filtered_df,
            ax=ax
        )

        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Exam Score")

        st.pyplot(fig)
        plt.close(fig)

      
        st.subheader("Class Attendance vs Exam Score")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.scatterplot(
            x="class_attendance",
            y="exam_score",
            data=filtered_df,
            ax=ax
        )

        ax.set_xlabel("Class Attendance (%)")
        ax.set_ylabel("Exam Score")

        st.pyplot(fig)
        plt.close(fig)

       
        st.subheader("Correlation Heatmap")

        numeric_df = filtered_df.select_dtypes(
            include="number"
        ).drop(columns=["student_id"], errors="ignore")

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)



elif page == "Model Comparison":

    st.title("Model Comparison")

    st.write(
        "Compare the performance of different regression "
        "models using evaluation metrics."
    )

    st.subheader("Model Performance")

    st.dataframe(
        comparison.round(3),
        use_container_width=True
    )

    st.subheader("Cross-Validation R2 Score Comparison")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        x="Model",
        y="CV R2 Score",
        data=comparison,
        ax=ax
    )

    ax.set_ylabel("Cross-Validation R2 Score")
    ax.set_xlabel("Regression Models")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=15)

    st.pyplot(fig)
    plt.close(fig)

  
    st.subheader("Test R2 Score Comparison")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        x="Model",
        y="R2 Score",
        data=comparison,
        ax=ax
    )

    ax.set_ylabel("Test R2 Score")
    ax.set_xlabel("Regression Models")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    
    st.subheader("Best Model")

    best_model_result = comparison.iloc[0]

    st.success(
        f"{best_model_result['Model']} achieved the highest "
        f"Cross-Validation R2 Score of "
        f"{best_model_result['CV R2 Score']:.3f}."
    )



elif page == "Prediction":

    st.title("Exam Score Prediction")

    st.write(
        "Enter student information to predict the expected exam score."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age", min_value=17, max_value=24, value=20
        )

        gender = st.selectbox(
            "Gender", ["male", "female", "other"]
        )

        course = st.selectbox(
            "Course",
            ["bca", "ba", "b.sc", "b.com", "bba", "diploma", "b.tech"]
        )

        study_hours = st.number_input(
            "Study Hours", min_value=0.08, max_value=7.91, value=4.0
        )

        class_attendance = st.number_input(
            "Class Attendance (%)",
            min_value=40.6, max_value=99.4, value=70.0
        )

        internet_access = st.selectbox(
            "Internet Access", ["yes", "no"]
        )

    with col2:

        sleep_hours = st.number_input(
            "Sleep Hours", min_value=4.1, max_value=9.9, value=7.0
        )

        sleep_quality = st.selectbox(
            "Sleep Quality", ["poor", "average", "good"]
        )

        study_method = st.selectbox(
            "Study Method",
            [
                "self-study",
                "online videos",
                "coaching",
                "group study",
                "mixed"
            ]
        )

        facility_rating = st.selectbox(
            "Facility Rating", ["low", "medium", "high"]
        )

        exam_difficulty = st.selectbox(
            "Exam Difficulty", ["easy", "moderate", "hard"]
        )

    if st.button("Predict Exam Score"):

        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "course": [course],
            "study_hours": [study_hours],
            "class_attendance": [class_attendance],
            "internet_access": [internet_access],
            "sleep_hours": [sleep_hours],
            "sleep_quality": [sleep_quality],
            "study_method": [study_method],
            "facility_rating": [facility_rating],
            "exam_difficulty": [exam_difficulty]
        })

        prediction = model.predict(input_data)

        st.success(
            f"Predicted Exam Score: {prediction[0]:.2f}"
        )
