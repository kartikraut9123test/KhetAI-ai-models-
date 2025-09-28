from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Crop Price API",
             description="API for accessing and analyzing crop price data")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class CropPrice(BaseModel):
    state: str
    district: str
    market: str
    crop: str
    variety: str
    date: datetime
    min_price: float
    max_price: float
    price_per_kg: float

class PricePrediction(BaseModel):
    crop: str
    predicted_price: float
    confidence: float

def load_data() -> pd.DataFrame:
    """Load the crop price data from CSV"""
    file_path = "datasets/crop_market_prices.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="No data available")
    return pd.read_csv(file_path, parse_dates=["date"])

@app.get("/prices/", response_model=List[CropPrice])
async def get_prices(
    crop: Optional[str] = Query(None, description="Filter by crop name"),
    state: Optional[str] = Query(None, description="Filter by state"),
    district: Optional[str] = Query(None, description="Filter by district"),
    market: Optional[str] = Query(None, description="Filter by market"),
    from_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """Get crop prices with optional filters"""
    try:
        df = load_data()
        
        # Apply filters
        if crop:
            df = df[df["crop"].str.contains(crop, case=False)]
        if state:
            df = df[df["state"].str.contains(state, case=False)]
        if district:
            df = df[df["district"].str.contains(district, case=False)]
        if market:
            df = df[df["market"].str.contains(market, case=False)]
        if from_date:
            df = df[df["date"] >= from_date]
        if to_date:
            df = df[df["date"] <= to_date]
            
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/")
async def get_stats(
    crop: str = Query(..., description="Crop name to get statistics for"),
    days: int = Query(30, description="Number of days to analyze")
):
    """Get price statistics for a specific crop"""
    try:
        df = load_data()
        df = df[df["crop"].str.contains(crop, case=False)]
        
        # Get recent data
        df = df.sort_values("date").tail(days)
        
        return {
            "crop": crop,
            "avg_price": float(df["price_per_kg"].mean()),
            "min_price": float(df["min_price"].min()),
            "max_price": float(df["max_price"].max()),
            "price_volatility": float(df["price_per_kg"].std()),
            "data_points": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
