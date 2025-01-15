from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib

# Load saved models and encoders
model = joblib.load('./model/random_forest_model.pkl')
imputer = joblib.load('./model/knn_imputer.pkl')
label_encoders = joblib.load('./model/encoders.pkl')

# Initialize FastAPI
app = FastAPI()

# Define the request body structure
class CarData(BaseModel):
    id: Optional[int] = None
    list_id: Optional[int] = None
    list_time: Optional[int] = None
    manufacture_date: Optional[int] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    origin: Optional[str] = None
    type: Optional[str] = None
    seats: Optional[float] = None
    gearbox: Optional[str] = None
    fuel: Optional[str] = None
    color: Optional[str] = None
    mileage_v2: Optional[float] = None
    price: Optional[float] = None
    condition: Optional[str] = None

@app.post("/predict/")
def predict_price(car_data: CarData):
    try:
        # Convert input to DataFrame
        data_dict = car_data.dict()
        df = pd.DataFrame([data_dict])

        # Encode categorical variables
        categorical_columns = ['brand', 'model', 'origin', 'type', 'gearbox', 'fuel', 'color', 'condition']
        for col in categorical_columns:
            encoder = label_encoders.get(col)
            if encoder:
                df[col] = df[col].fillna('missing')  # Placeholder for missing values
                df[col] = encoder.transform(df[col])
                # Revert placeholder back to NaN
                missing_encoded = encoder.transform(['missing'])[0]
                df[col] = df[col].replace(missing_encoded, np.nan)
            else:
                raise HTTPException(status_code=400, detail=f"No encoder found for column '{col}'")

        # Drop unnecessary columns
        df.drop(columns=["id", "list_id", "list_time", "price", "condition"], inplace=True)

        # Handle missing values with KNN imputer
        missing_values_mask = df.isnull()
        df_imputed = imputer.transform(df)
        df_numpy = np.where(missing_values_mask, df_imputed, df)
        df_numpy = df_numpy.astype(float)
        df = pd.DataFrame(df_numpy, columns=df.columns)

        # Log transformations
        df['mileage_v2'] = np.log(df['mileage_v2'] + 0.0001)

        # Add 'car_age' feature
        df['car_age'] = 2025 - df['manufacture_date']
        df.drop(columns=['manufacture_date'], inplace=True)
        df['car_age'] = np.log(df['car_age'] + 0.0001)

        # Align with the model's expected features
        expected_features = model.feature_names_in_
        df = df[expected_features]

        # Predict price
        predicted_price = model.predict(df)
        predicted_price = predicted_price * 1e6  # Reverse scaling

        return {"predicted_price": predicted_price[0]}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing or unexpected feature: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API"}
