import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings('ignore')

# Use docker-compose service name because we train inside the container
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_and_prepare_data():
    df = pd.read_csv("student_exam_scores.csv", sep=',')
    target = 'exam_score'
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y

def train_and_register_model(model_name="StudentExamPredictor"):
    mlflow.set_experiment("Student_Academic_Trends_Docker")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🔗 Run ID: {run_id}")

        X, y = load_and_prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model + training
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=2, random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions + metrics
        y_pred = model.predict(X_test)
        mse, rmse, r2 = mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)

        mlflow.log_metrics({"mse": mse, "rmse": rmse, "r2_score": r2})

        # Log model into artifacts
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:3],
            registered_model_name=model_name,
        )

        print(f"✅ Model registered in MLflow as '{model_name}'")
        print(f"   Metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        return model, run_id

if __name__ == "__main__":
    print("🐳 Training + registering model inside Docker container...")
    train_and_register_model()
