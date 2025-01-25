import pandas as pd
import numpy as np
import psycopg2
from config import DB_CONFIG

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: (MACD line, Signal line, MACD histogram)
    """
    # Calculate EMAs
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = exp1 - exp2
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    """
    return data.ewm(span=period, adjust=False).mean()
           
def get_technical_signals(df: pd.DataFrame, currency: str = None) -> pd.DataFrame:
    """
    Calculate technical signals and generate buy/sell indicators
    """
    df = df.copy()
    
    # Calculate RSI
    print("Calculating RSI...")
    df['rsi'] = calculate_rsi(df['price_usd'])
    
    # Calculate MACD
    print("Calculating MACD...")
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['price_usd'])
    
    # Calculate EMAs
    print("Calculating EMAs...")
    df['ema_50'] = calculate_ema(df['price_usd'], 50)
    df['ema_200'] = calculate_ema(df['price_usd'], 200)
    
    # Generate signals
    print("Generating trading signals...")
    
    # RSI signals - adding explicit boolean conversion
    df['rsi_oversold'] = (df['rsi'] < 30).astype(bool)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(bool)
    
    # MACD signals - adding explicit boolean conversion
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & 
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(bool)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) & 
                            (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(bool)
    
    # EMA signals - adding explicit boolean conversion
    df['ema_bullish'] = (df['ema_50'] > df['ema_200']).astype(bool)
    df['ema_bearish'] = (df['ema_50'] < df['ema_200']).astype(bool)
    df['golden_cross'] = ((df['ema_50'] > df['ema_200']) & 
                         (df['ema_50'].shift(1) <= df['ema_200'].shift(1))).astype(bool)
    df['death_cross'] = ((df['ema_50'] < df['ema_200']) & 
                        (df['ema_50'].shift(1) >= df['ema_200'].shift(1))).astype(bool)
    
    # Combined signals - adding explicit boolean conversion
    df['buy_signal'] = (
        (df['macd_crossover'] & df['rsi_oversold']) |  # MACD crosses up while RSI oversold
        (df['golden_cross'] & (df['rsi'] < 60)) |      # Golden cross with reasonable RSI
        (df['rsi_oversold'] & df['ema_bullish'])       # Oversold in bullish trend
    ).astype(bool)
    
    df['sell_signal'] = (
        (df['macd_crossunder'] & df['rsi_overbought']) |  # MACD crosses down while RSI overbought
        (df['death_cross'] & (df['rsi'] > 40)) |          # Death cross with reasonable RSI
        (df['rsi_overbought'] & df['ema_bearish'])        # Overbought in bearish trend
    ).astype(bool)

    # Add validation prints
    print(f"Signal counts:")
    print(f"Buy signals: {df['buy_signal'].sum()}")
    print(f"Sell signals: {df['sell_signal'].sum()}")
    print(f"Golden crosses: {df['golden_cross'].sum()}")
    print(f"Death crosses: {df['death_cross'].sum()}")

    # Save technical indicators to database if currency is provided
    if currency:
        print(f"Saving technical indicators for {currency}...")
        save_technical_indicators(df, currency)
    
    return df

def save_technical_indicators(df: pd.DataFrame, currency: str):
    """
    Save technical indicators to database with improved boolean handling
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Create table if it doesn't exist
        cur.execute("""
        CREATE TABLE IF NOT EXISTS crypto_analysis (
            id SERIAL PRIMARY KEY,
            currency VARCHAR(20),
            date DATE,
            rsi NUMERIC,
            macd NUMERIC,
            macd_signal NUMERIC,
            macd_hist NUMERIC,
            ema_50 NUMERIC,
            ema_200 NUMERIC,
            buy_signal BOOLEAN,
            sell_signal BOOLEAN,
            golden_cross BOOLEAN,
            death_cross BOOLEAN,
            UNIQUE(currency, date)
        )
        """)

        # Prepare data for insertion
        records_processed = 0
        for idx, row in df.iterrows():
            # Convert boolean values explicitly
            buy_signal = bool(row['buy_signal'])
            sell_signal = bool(row['sell_signal'])
            golden_cross = bool(row['golden_cross'])
            death_cross = bool(row['death_cross'])

            cur.execute("""
            INSERT INTO crypto_analysis (
                currency, date, rsi, macd, macd_signal, macd_hist,
                ema_50, ema_200, buy_signal, sell_signal,
                golden_cross, death_cross
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (currency, date) 
            DO UPDATE SET 
                rsi = EXCLUDED.rsi,
                macd = EXCLUDED.macd,
                macd_signal = EXCLUDED.macd_signal,
                macd_hist = EXCLUDED.macd_hist,
                ema_50 = EXCLUDED.ema_50,
                ema_200 = EXCLUDED.ema_200,
                buy_signal = EXCLUDED.buy_signal,
                sell_signal = EXCLUDED.sell_signal,
                golden_cross = EXCLUDED.golden_cross,
                death_cross = EXCLUDED.death_cross
            """, (
                currency,
                idx.date(),
                float(row['rsi']) if not pd.isna(row['rsi']) else None,
                float(row['macd']) if not pd.isna(row['macd']) else None,
                float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,
                float(row['macd_hist']) if not pd.isna(row['macd_hist']) else None,
                float(row['ema_50']) if not pd.isna(row['ema_50']) else None,
                float(row['ema_200']) if not pd.isna(row['ema_200']) else None,
                buy_signal,
                sell_signal,
                golden_cross,
                death_cross
            ))
            records_processed += 1
            
            # Commit every 100 records and print progress
            if records_processed % 100 == 0:
                conn.commit()
                print(f"Processed {records_processed} records for {currency}")

        conn.commit()
        print(f"Successfully saved {records_processed} technical indicators for {currency}")
        
        # Validate the saved data
        cur.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN buy_signal THEN 1 ELSE 0 END) as buy_signals,
            SUM(CASE WHEN sell_signal THEN 1 ELSE 0 END) as sell_signals,
            SUM(CASE WHEN golden_cross THEN 1 ELSE 0 END) as golden_crosses,
            SUM(CASE WHEN death_cross THEN 1 ELSE 0 END) as death_crosses
        FROM crypto_analysis
        WHERE currency = %s
        """, (currency,))
        
        stats = cur.fetchone()
        print("\nValidation of saved data:")
        print(f"Total records: {stats[0]}")
        print(f"Buy signals: {stats[1]}")
        print(f"Sell signals: {stats[2]}")
        print(f"Golden crosses: {stats[3]}")
        print(f"Death crosses: {stats[4]}")

    except Exception as e:
        print(f"Error saving technical indicators: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def backtest_signals(df: pd.DataFrame, initial_capital: float = 10000.0) -> tuple:
    """
    Backtest trading signals and calculate performance metrics
    Returns: (DataFrame with results, Dict with metrics)
    """
    df = df.copy()
    
    # Initialize position and portfolio columns
    df['position'] = 0  # 1 for long, 0 for neutral
    df['portfolio_value'] = initial_capital
    df['holdings'] = 0.0
    df['cash'] = initial_capital
    
    # Trading simulation
    current_position = 0
    
    for i in range(1, len(df)):
        # Update position based on signals
        if df['buy_signal'].iloc[i] and current_position == 0:
            # Buy signal
            current_position = 1
            df.loc[df.index[i], 'position'] = 1
            # Calculate holdings
            price = df['price_usd'].iloc[i]
            cash_available = df['cash'].iloc[i-1]
            units = cash_available / price
            df.loc[df.index[i], 'holdings'] = units * price
            df.loc[df.index[i], 'cash'] = cash_available - (units * price)
            
        elif df['sell_signal'].iloc[i] and current_position == 1:
            # Sell signal
            current_position = 0
            df.loc[df.index[i], 'position'] = 0
            # Calculate cash after selling
            holdings = df['holdings'].iloc[i-1]
            df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] + holdings
            df.loc[df.index[i], 'holdings'] = 0
            
        else:
            # No change in position
            df.loc[df.index[i], 'position'] = current_position
            df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1]
            if current_position == 1:
                # Update holdings value
                price_change = df['price_usd'].iloc[i] / df['price_usd'].iloc[i-1]
                df.loc[df.index[i], 'holdings'] = df['holdings'].iloc[i-1] * price_change
            else:
                df.loc[df.index[i], 'holdings'] = 0
                
        # Update portfolio value
        df.loc[df.index[i], 'portfolio_value'] = df['holdings'].iloc[i] + df['cash'].iloc[i]
    
    # Calculate performance metrics
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate daily returns
    df['daily_returns'] = df['portfolio_value'].pct_change()
    
    # Calculate metrics
    sharpe_ratio = np.sqrt(252) * (df['daily_returns'].mean() / df['daily_returns'].std())
    max_drawdown = (df['portfolio_value'] / df['portfolio_value'].cummax() - 1).min()
    
    # Count trades
    buy_signals = df['buy_signal'].sum()
    sell_signals = df['sell_signal'].sum()
    
    # Win rate
    profitable_trades = ((df['portfolio_value'] > df['portfolio_value'].shift(1)) & 
                        (df['position'] != df['position'].shift(1))).sum()
    total_trades = buy_signals  # or sell_signals, should be equal
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'number_of_trades': total_trades,
        'win_rate': win_rate,
        'final_value': final_value,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals
    }
    
    return df, metrics