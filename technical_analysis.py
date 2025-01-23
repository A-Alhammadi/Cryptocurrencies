import pandas as pd
import numpy as np

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

def get_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical signals and generate buy/sell indicators
    """
    df = df.copy()
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df['price_usd'])
    
    # Calculate MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['price_usd'])
    
    # Calculate EMAs
    df['ema_50'] = calculate_ema(df['price_usd'], 50)
    df['ema_200'] = calculate_ema(df['price_usd'], 200)
    
    # Generate signals
    
    # RSI signals
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_overbought'] = df['rsi'] > 70
    
    # MACD signals
    df['macd_crossover'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_crossunder'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # EMA signals
    df['ema_bullish'] = df['ema_50'] > df['ema_200']
    df['ema_bearish'] = df['ema_50'] < df['ema_200']
    df['golden_cross'] = (df['ema_50'] > df['ema_200']) & (df['ema_50'].shift(1) <= df['ema_200'].shift(1))
    df['death_cross'] = (df['ema_50'] < df['ema_200']) & (df['ema_50'].shift(1) >= df['ema_200'].shift(1))
    
    # Combined signals
    df['buy_signal'] = (
        (df['macd_crossover'] & df['rsi_oversold']) |  # MACD crosses up while RSI oversold
        (df['golden_cross'] & (df['rsi'] < 60)) |      # Golden cross with reasonable RSI
        (df['rsi_oversold'] & df['ema_bullish'])       # Oversold in bullish trend
    )
    
    df['sell_signal'] = (
        (df['macd_crossunder'] & df['rsi_overbought']) |  # MACD crosses down while RSI overbought
        (df['death_cross'] & (df['rsi'] > 40)) |          # Death cross with reasonable RSI
        (df['rsi_overbought'] & df['ema_bearish'])        # Overbought in bearish trend
    )
    
    return df

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
    
    return df, metrics  # Return both the DataFrame and the metrics