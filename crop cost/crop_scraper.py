# crop_scraper.py
import requests
import pandas as pd
import datetime
import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

def fetch_crop_prices(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch crop prices from data.gov.in API
    
    Args:
        api_key (str, optional): API key for data.gov.in. If not provided, will look for API_KEY in environment variables.
    
    Returns:
        pd.DataFrame: DataFrame containing crop price data
    """
    BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    
    # Get API key from environment variable if not provided
    if not api_key:
        try:
            with open('.env', 'r') as f:
                content = f.read().strip()
                api_key = content.split('=')[1].strip()
                if api_key == 'YOUR_API_KEY':
                    raise ValueError("Please replace YOUR_API_KEY with your actual API key in the .env file")
        except Exception as e:
            print(f"Error reading .env file: {e}")
            api_key = os.getenv('API_KEY')
            
        if not api_key:
            raise ValueError("API key not found. Please set API_KEY environment variable or provide it as an argument.")
            
    print(f"Using API key: {api_key[:10]}...")
    
    all_records = []
    offset = 0
    page_size = 10000  # Maximum allowed by the API
    total_records = None
    
    while True:
        params = {
            'api-key': api_key,
            'format': 'json',
            'limit': 100,  # Reduced batch size
            'offset': offset
        }
        
        print(f"\nFetching records {offset} to {offset + 100}...")
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code != 200:
                print(f"Error: Failed to fetch crop prices. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                if len(all_records) > 0:
                    break  # Use what we have if we've got some records
                sys.exit(1)
                
            data = response.json()
            if 'records' not in data:
                print(f"Error: Unexpected API response format: {data}")
                if len(all_records) > 0:
                    break  # Use what we have if we've got some records
                sys.exit(1)
            
            records = data['records']
            if not records:  # If no records returned
                break
                
            all_records.extend(records)
            
            # Get total count from first response
            if total_records is None and 'total' in data:
                total_records = int(data['total'])
                print(f"\nTotal available records: {total_records}")
            
            offset += len(records)  # Increment by actual number of records received
            
            # Show progress
            print(f"Progress: {len(all_records)}/{total_records if total_records else 'unknown'} records")
            
            # Add a small delay to avoid overwhelming the API
            import time
            time.sleep(0.5)  # 500ms delay between requests
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            if len(all_records) > 0:
                break  # Use what we have if we've got some records
            sys.exit(1)
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            break
    
    print(f"\nTotal records fetched: {len(all_records)}")
    
    # Convert all records into DataFrame
    df = pd.DataFrame(all_records)
    
    # Normalize column names
    df = df.rename(columns={
        "state": "state",
        "district": "district",
        "market": "market",
        "commodity": "crop",
        "variety": "variety",
        "arrival_date": "date",
        "min_price": "min_price",
        "max_price": "max_price",
        "modal_price": "price_per_kg"
    })

    # Keep required columns and convert data types
    df = df[["state", "district", "market", "crop", "variety", "date", "min_price", "max_price", "price_per_kg"]]
    
    # Convert date and numeric columns
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    numeric_columns = ["min_price", "max_price", "price_per_kg"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    return df

def update_crop_prices(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch and save crop prices to CSV
    
    Args:
        api_key (str, optional): API key for data.gov.in
        
    Returns:
        pd.DataFrame: DataFrame containing the fetched data
    """
    try:
        df = fetch_crop_prices(api_key)
        os.makedirs("datasets", exist_ok=True)
        
        # Save to CSV with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"datasets/crop_market_prices_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\nData saved to: {filename}")
        return df
    except Exception as e:
        print(f"Error updating crop prices: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        df = update_crop_prices()
        if df is not None:
            print("\nSuccessfully updated crop prices.")
            print("\nFirst few entries:")
            print(df.head())
            print(f"\nTotal records fetched: {len(df)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
