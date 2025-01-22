import psycopg2
import pandas as pd
from datetime import datetime

def load_data_from_db(db_config: dict, currency_symbol: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Load cryptocurrency data from database with proper date handling
    """
    print(f"\nDatabase connection attempt:")
    print(f"Host: {db_config['host']}")
    print(f"Database: {db_config['dbname']}")
    print(f"Currency: {currency_symbol}")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        conn = psycopg2.connect(**db_config)
        query = """
        SELECT date::date as date, price_usd, volume_usd
        FROM crypto_prices
        WHERE currency = %s
        """
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
        query += " ORDER BY date ASC;"
        
        params = [currency_symbol]
        if start_date and end_date:
            params.extend([start_date, end_date])
        
        # Load data    
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index but keep it as a column too
        df.set_index('date', inplace=True, drop=False)
        
        print(f"Data loaded successfully for {currency_symbol}")
        print(f"Date range in data: {df.index.min()} to {df.index.max()}")
        print(f"Number of rows: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading {currency_symbol}: {str(e)}")
        return pd.DataFrame()