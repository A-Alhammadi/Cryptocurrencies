import yfinance as yf
import pandas as pd
import psycopg2
from datetime import datetime
from config import DB_CONFIG, CURRENCIES

def setup_database():
    # Connect to default database first to create our database if needed
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host']
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # Create database if it doesn't exist
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
    exists = cur.fetchone()
    if not exists:
        cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
    
    cur.close()
    conn.close()
    
    # Connect to our database and create table
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Create table with all available fields from Yahoo Finance
    cur.execute("""
    DROP TABLE IF EXISTS crypto_prices;
    CREATE TABLE crypto_prices (
        id SERIAL PRIMARY KEY,
        currency VARCHAR(20),
        date DATE,
        open_price NUMERIC,
        high_price NUMERIC,
        low_price NUMERIC,
        close_price NUMERIC,
        volume NUMERIC,
        UNIQUE(currency, date)
    )
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database setup completed")

def process_currency(currency):
    try:
        # Download data
        print(f"Downloading data for {currency}...")
        data = yf.download(currency, progress=False)
        
        if data.empty:
            print(f"No data found for {currency}")
            return
        
        # Connect to the database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Convert index to datetime if it isn't already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        print(f"Processing {len(data)} records for {currency}")
        
        # Process each row
        records_processed = 0
        
        for date_idx, row in data.iterrows():
            try:
                insert_query = """
                INSERT INTO crypto_prices 
                    (currency, date, open_price, high_price, low_price, close_price, volume)
                VALUES 
                    (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (currency, date) 
                DO UPDATE SET 
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume
                """
                
                values = (
                    currency,
                    date_idx.date(),
                    float(row[('Open', currency)]),
                    float(row[('High', currency)]),
                    float(row[('Low', currency)]),
                    float(row[('Close', currency)]),
                    float(row[('Volume', currency)])
                )
                
                cur.execute(insert_query, values)
                records_processed += 1
                
                # Commit every 100 records
                if records_processed % 100 == 0:
                    conn.commit()
                    print(f"Processed {records_processed} records for {currency}")
                
            except Exception as e:
                print(f"Error processing record for {currency} on {date_idx.date()}: {str(e)}")
                continue
        
        # Final commit
        conn.commit()
        print(f"Successfully processed {records_processed} records for {currency}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error processing {currency}: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()

def main():
    print("Setting up database...")
    setup_database()
    
    for currency in CURRENCIES:
        print(f"\nProcessing {currency}...")
        process_currency(currency)
    
    print("\nData retrieval completed!")

if __name__ == "__main__":
    main()