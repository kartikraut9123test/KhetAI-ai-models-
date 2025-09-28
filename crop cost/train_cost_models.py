import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import glob
import os
from datetime import datetime
import calendar
from crop_scraper import update_crop_prices

def load_latest_dataset():
    """Load the most recent crop price dataset"""
    # Get the most recent file
    files = glob.glob("datasets/crop_market_prices_*.csv")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading dataset: {latest_file}")
    return pd.read_csv(latest_file, parse_dates=["date"])

def prepare_features(df):
    """Prepare features for model training with advanced feature engineering"""
    # Extract temporal features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    
    # Add seasonal features
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9])  # Indian monsoon season
    df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                    else 'Spring' if x in [3, 4, 5]
                                    else 'Summer' if x in [6, 7, 8]
                                    else 'Fall')
    
    # Add market demand indicators
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_month_start'] = df['day_of_month'] <= 5
    df['is_month_end'] = df['day_of_month'] >= 25
    
    # Calculate rolling statistics for each crop and market
    df = df.sort_values('date')
    
    # Group by crop and market
    for group_cols in [['crop'], ['market'], ['crop', 'market']]:
        group = df.groupby(group_cols)
        
        # Calculate rolling means with different windows
        for window in [7, 14, 30]:
            col_suffix = '_'.join(group_cols + [f'{window}d'])
            df[f'price_mean_{col_suffix}'] = group['price_per_kg'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'price_std_{col_suffix}'] = group['price_per_kg'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
    
    # Encode categorical variables with memory of previous encoding
    encoders = {}
    for col in ['state', 'district', 'market', 'crop', 'variety', 'season']:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col])
    
    # Save encoders for later use
    os.makedirs('models', exist_ok=True)
    for col, encoder in encoders.items():
        joblib.dump(encoder, f'models/{col}_encoder.joblib')
    
    # Create feature matrix
    features = [
        # Categorical encodings
        'state_encoded', 'district_encoded', 'market_encoded',
        'crop_encoded', 'variety_encoded',
        
        # Temporal features
        'month', 'year', 'day_of_week', 'day_of_month', 'quarter',
        
        # Seasonal indicators
        'is_monsoon',
        
        # Market indicators
        'is_weekend', 'is_month_start', 'is_month_end'
    ]
    
    # Add rolling statistics features
    rolling_features = [col for col in df.columns if 'price_' in col and col != 'price_per_kg']
    features.extend(rolling_features)
    
    # Create feature matrix and handle missing values
    X = df[features].copy()
    X = X.fillna(method='ffill').fillna(method='bfill')  # Forward-fill then backward-fill any NaN
    
    return X, df['price_per_kg']

def evaluate_model(model, X, y, scaler, model_name="Model"):
    """Evaluate model performance with multiple metrics"""
    # Make predictions
    y_pred = model.predict(scaler.transform(X))
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}\n")
    
    return r2, rmse, mae

def train_models(X, y):
    """Train and evaluate multiple models with cross-validation and hyperparameter tuning"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'models/feature_scaler.joblib')
    
    # 1. Random Forest with GridSearch CV
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    rf_model = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    print("Training Random Forest model with cross-validation...")
    rf_model.fit(X_train_scaled, y_train)
    print(f"Best RF parameters: {rf_model.best_params_}")
    
    # 2. XGBoost with GridSearch CV
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9]
    }
    
    xgb_model = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        xgb_params,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    print("\nTraining XGBoost model with cross-validation...")
    xgb_model.fit(X_train_scaled, y_train)
    print(f"Best XGB parameters: {xgb_model.best_params_}")
    
    # Evaluate models on test set
    print("\nEvaluating models on test set:")
    rf_scores = evaluate_model(rf_model, X_test, y_test, scaler, "Random Forest")
    xgb_scores = evaluate_model(xgb_model, X_test, y_test, scaler, "XGBoost")
    
    # Select the best model based on test R²
    best_model = rf_model if rf_scores[0] > xgb_scores[0] else xgb_model
    model_name = "Random Forest" if rf_scores[0] > xgb_scores[0] else "XGBoost"
    
    print(f"\nBest performing model: {model_name}")
    
    # Save the best model
    joblib.dump(best_model, 'models/crop_price_model.joblib')
    
    # Feature importance analysis
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importances.head(10))
    
    return best_model, scaler

if __name__ == "__main__":
    # First, get latest data from API
    try:
        print("Fetching latest crop prices...")
        new_crops = update_crop_prices()
        print(f"New crop prices added: {new_crops.shape[0]} records")
    except Exception as e:
        print(f"Warning: Could not fetch new data: {str(e)}")
    
    # Load the dataset
    print("\nLoading dataset...")
    crop_df = load_latest_dataset()
    
    if crop_df is None:
        print("Error: No dataset found. Please run crop_scraper.py first.")
        exit(1)
    
    print(f"Dataset loaded: {crop_df.shape[0]} records")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = prepare_features(crop_df)
    
    print("\nFeature set summary:")
    print(f"Number of features: {X.shape[1]}")
    print("Features:", ", ".join(X.columns))
    
    # Train and evaluate models
    model, scaler = train_models(X, y)
    
    print("\nModel training completed!")
    print("Model and preprocessing objects saved in the 'models' directory.")
