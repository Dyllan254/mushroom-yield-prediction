import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import logging
from fastapi.responses import JSONResponse
import os
import uvicorn

# ----------- MODEL TRAINING AND SAVING --------------

# Step 1: Load the dataset
df = pd.read_csv('mushroom_harvest_cycles_with_nulls.csv')

# Step 2: Handle missing values (impute with mean)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])
df_clean = pd.concat([df['Harvest Cycle'], df_imputed], axis=1)

# Step 3: Define features and target
X = df_clean.drop(['Yield (kg)', 'Harvest Cycle'], axis=1)
y = df_clean['Yield (kg)']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Hyperparameter tuning

# Grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='r2')
grid_search_rf.fit(X_train_scaled, y_train)
best_rf = grid_search_rf.best_estimator_
print("🔍 Best Random Forest params:", grid_search_rf.best_params_)

# Grid for SVR
param_grid_svr = {
    'C': [1, 10],
    'epsilon': [0.1, 0.5],
    'kernel': ['rbf']
}
grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=3, scoring='r2')
grid_search_svr.fit(X_train_scaled, y_train)
best_svr = grid_search_svr.best_estimator_
print("🔍 Best SVR params:", grid_search_svr.best_params_)

# Step 7: Define base models and meta model using the tuned models
base_models = [
    ('rf', best_rf),
    ('svr', best_svr),
    ('lr', LinearRegression())
]
meta_model = LinearRegression()

# Step 8: Evaluate base models
print("\n📊 Base Model Predictions:")
for name, model in base_models:
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    print(f"\n{name.upper()} Results:")
    print(f"  MAE: {mean_absolute_error(y_test, pred):.2f}")
    print(f"  MSE: {mean_squared_error(y_test, pred):.2f}")
    print(f"  R² : {r2_score(y_test, pred):.4f}")

# Step 9: Define and train stacking regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train_scaled, y_train)
y_pred = stacking_model.predict(X_test_scaled)

print("\n📈 Stacking Model Test Set Results:")
print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"  MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"  R² : {r2_score(y_test, y_pred):.4f}")

# Step 10: Cross-validation
pipeline = make_pipeline(StandardScaler(), stacking_model)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

print("\n🔁 Cross-Validation Results (Stacking Model):")
print(f"  R² Scores (folds): {cv_scores}")
print(f"  Average R² Score : {np.mean(cv_scores):.4f}")

# Step 11: Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Yield (kg)")
plt.ylabel("Predicted Yield (kg)")
plt.title("Actual vs Predicted Yield (Stacking Model)")
plt.grid(True)
plt.show()

# Save model and scaler
joblib.dump(stacking_model, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# ----------- FASTAPI APP --------------

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class MushroomInput(BaseModel):
    temperature: float
    humidity: float
    ph: float
    co2: float

# JSON-based prediction endpoint
@app.post("/predict")
def predict(data: MushroomInput):
    try:
        logger.info("Received JSON prediction request: %s", data)
        input_data = np.array([[data.temperature, data.humidity, data.ph, data.co2]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = {"predicted_yield_kg": round(float(prediction[0]), 2)}
        logger.info("Prediction result: %s", result)
        return result
    except Exception as e:
        logger.error("Error during JSON prediction: %s", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal server error during prediction."})

# CSV-based prediction endpoint
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        logger.info("Received file: %s", file.filename)

        # Restrict file format to .csv only
        if not file.filename.endswith('.csv'):
            logger.warning("Rejected file: %s (Invalid format)", file.filename)
            return JSONResponse(status_code=400, content={"error": "Only CSV files are accepted."})

        # Read CSV file into DataFrame
        df = pd.read_csv(file.file)

        # Ensure correct columns are present
        required_columns = ['temperature', 'humidity', 'ph', 'co2']
        if not all(col in df.columns for col in required_columns):
            logger.error("CSV missing required columns: %s", df.columns.tolist())
            return JSONResponse(
                status_code=400,
                content={"error": f"CSV must contain the following columns: {required_columns}"}
            )

        # Predict
        input_scaled = scaler.transform(df[required_columns])
        predictions = model.predict(input_scaled)
        df["predicted_yield_kg"] = [round(float(pred), 2) for pred in predictions]

        logger.info("Successfully predicted %d rows", len(predictions))
        return df.to_dict(orient="records")

    except Exception as e:
        logger.error("Error during CSV prediction: %s", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal server error during CSV prediction."})

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Mushroom Yield Prediction"}



