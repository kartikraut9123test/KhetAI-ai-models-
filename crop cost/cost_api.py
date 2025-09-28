from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI(
    title="Crop Price Prediction API",
    description="API for accessing historical crop prices and predicting future prices",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and preprocessors
try:
    model = joblib.load('models/crop_price_model.joblib')
    scaler = joblib.load('models/feature_scaler.joblib')
    
    # Load encoders
    encoders = {}
    for col in ['state', 'district', 'market', 'crop', 'variety']:  # Removed 'season' as it's not used
        encoder_path = f'models/{col}_encoder.joblib'
        if os.path.exists(encoder_path):
            encoders[col] = joblib.load(encoder_path)
        else:
            print(f"Warning: Encoder not found for {col}")
    
    # Validate required encoders
    required_encoders = ['state', 'district', 'market', 'crop', 'variety']
    missing_encoders = [col for col in required_encoders if col not in encoders]
    if missing_encoders:
        raise ValueError(f"Missing required encoders: {missing_encoders}")
        
except Exception as e:
    print(f"Error loading model and preprocessors: {str(e)}")
    model = None
    scaler = None
    encoders = {}

# Pydantic models for request/response
class PricePredictionRequest(BaseModel):
    state: str
    district: str
    market: str
    crop: str
    variety: str
    date: str

class PricePrediction(BaseModel):
    predicted_price: float
    confidence_interval: dict
    historical_trend: dict

def load_latest_data():
    """Load the most recent crop price dataset"""
    files = glob.glob("datasets/crop_market_prices_*.csv")
    if not files:
        raise HTTPException(status_code=404, detail="No price data available")
    latest_file = max(files, key=os.path.getctime)
    return pd.read_csv(latest_file, parse_dates=["date"])

def prepare_features(data):
    """Prepare features for prediction"""
    df = data.copy()
    
    # Extract temporal features
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['day_of_month'] = pd.to_datetime(df['date']).dt.day
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    
    # Add seasonal features
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9])
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_month_start'] = df['day_of_month'] <= 5
    df['is_month_end'] = df['day_of_month'] >= 25
    
    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[f'{col}_encoded'] = encoder.transform(df[col])
            except:
                # Handle unknown categories
                df[f'{col}_encoded'] = -1
    
    # Select features in the same order as training
    feature_cols = [
        'state_encoded', 'district_encoded', 'market_encoded',
        'crop_encoded', 'variety_encoded', 'month', 'year',
        'day_of_week', 'day_of_month', 'quarter', 'is_monsoon',
        'is_weekend', 'is_month_start', 'is_month_end'
    ]
    
    return df[feature_cols]

@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Crop Price Prediction API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/crop-prices")
async def get_latest_crop_prices(
    crop: Optional[str] = None,
    state: Optional[str] = None,
    days: Optional[int] = 30
):
    """Get latest crop prices with optional filtering"""
    try:
        df = load_latest_data()
        
        if crop:
            df = df[df['crop'].str.contains(crop, case=False)]
        if state:
            df = df[df['state'].str.contains(state, case=False)]
            
        # Get recent data
        latest_date = df['date'].max()
        cutoff_date = pd.to_datetime(latest_date) - pd.Timedelta(days=days)
        df = df[pd.to_datetime(df['date']) >= cutoff_date]
        
        # Calculate statistics
        stats = df.groupby('crop').agg({
            'price_per_kg': ['mean', 'min', 'max', 'std'],
            'date': 'max'
        }).reset_index()
        
        stats.columns = ['crop', 'avg_price', 'min_price', 'max_price', 'price_std', 'last_update']
        return JSONResponse(content=stats.to_dict(orient="records"))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_price(request: PricePredictionRequest):
    """Predict crop price for given parameters"""
    if not model or not scaler:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure the model has been trained."
        )
    
    if not all(col in encoders for col in ['state', 'district', 'market', 'crop', 'variety']):
        raise HTTPException(
            status_code=503,
            detail="Required encoders not loaded. Please ensure the model has been trained."
        )
    
    try:
        # Validate input date
        try:
            pred_date = pd.to_datetime(request.date)
        except:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Please use YYYY-MM-DD format."
            )
        
        # Create prediction data
        pred_data = pd.DataFrame([{
            'state': request.state,
            'district': request.district,
            'market': request.market,
            'crop': request.crop,
            'variety': request.variety,
            'date': pred_date
        }])
        
        # Validate categorical inputs
        for col in ['state', 'district', 'market', 'crop', 'variety']:
            if pred_data[col].iloc[0] not in encoders[col].classes_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown {col}: {pred_data[col].iloc[0]}. Available values: {', '.join(encoders[col].classes_)}"
                )
        
        # Prepare features
        try:
            X = prepare_features(pred_data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error preparing features: {str(e)}"
            )
        
        # Scale features
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error scaling features: {str(e)}"
            )
        
        # Make prediction
        try:
            predicted_price = float(model.predict(X_scaled)[0])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error making prediction: {str(e)}"
            )
        
        # Get historical trend
        try:
            df = load_latest_data()
            mask = (
                (df['crop'] == request.crop) &
                (df['market'] == request.market)
            )
            history = df[mask].sort_values('date').tail(30)
            
            trend = {
                'dates': history['date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': history['price_per_kg'].tolist()
            }
            
            # Calculate confidence interval (using historical std dev)
            std_dev = history['price_per_kg'].std() or predicted_price * 0.1  # Use 10% if no history
            confidence_interval = {
                'lower': max(0, predicted_price - 1.96 * std_dev),  # Ensure non-negative
                'upper': predicted_price + 1.96 * std_dev
            }
        except Exception as e:
            # If historical data fails, provide simplified confidence interval
            trend = {'dates': [], 'prices': []}
            confidence_interval = {
                'lower': max(0, predicted_price * 0.9),
                'upper': predicted_price * 1.1
            }
        
        return PricePrediction(
            predicted_price=predicted_price,
            confidence_interval=confidence_interval,
            historical_trend=trend
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/markets")
async def get_markets(state: Optional[str] = None):
    """Get list of markets, optionally filtered by state"""
    try:
        df = load_latest_data()
        if state:
            df = df[df['state'].str.contains(state, case=False)]
        
        markets = df.groupby(['state', 'district', 'market']).size().reset_index()
        markets = markets.drop(columns=[0])
        return JSONResponse(content=markets.to_dict(orient="records"))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crops")
async def get_crops():
    """Get list of available crops with their varieties"""
    try:
        df = load_latest_data()
        crops = df.groupby(['crop', 'variety']).size().reset_index()
        crops = crops.drop(columns=[0])
        return JSONResponse(content=crops.to_dict(orient="records"))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
