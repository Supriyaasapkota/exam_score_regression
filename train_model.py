
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("exam_score_cleaned.csv")



X = df.drop(columns=["exam_score"])
y = df["exam_score"]



numeric_features = X.select_dtypes(include="number").columns.tolist()

categorical_features = X.select_dtypes(
    include="str"
).columns.tolist()



preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



models = {
    "Linear Regression": (
        LinearRegression(),
        {}
    ),

    "KNN Regression": (
        KNeighborsRegressor(),
        {
            "model__n_neighbors": [3, 5, 7, 9]
        }
    ),

    "Support Vector Regression": (
        SVR(),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    )
}


results = []

best_model = None
best_model_name = None
best_cv_r2 = None



for name, (estimator, param_grid) in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    if param_grid:

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_

        cv_r2 = grid_search.best_score_

        print("Best Parameters:", grid_search.best_params_)

    else:

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        cv_r2 = cv_scores.mean()

        model = pipeline
        model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

   
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "CV R2 Score": cv_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("Cross-Validation R2 Score:", round(cv_r2, 4))
    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("Test R2 Score:", round(r2, 4))

    
    if best_cv_r2 is None or cv_r2 > best_cv_r2:

        best_cv_r2 = cv_r2
        best_model = model
        best_model_name = name


comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="CV R2 Score",
    ascending=False
)

comparison.to_csv("model_comparison.csv", index=False)



joblib.dump(best_model, "best_model.joblib")


print("\nModel Comparison:")
print(comparison)

print("\nBest Model:", best_model_name)
print("Best Cross-Validation R2 Score:", round(best_cv_r2, 4))

print("\nBest Model Saved Successfully!")
