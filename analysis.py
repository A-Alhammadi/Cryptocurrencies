import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def detect_cycles(price_series, sampling_period=1):
    """
    Detect dominant cycles using FFT and spectral analysis
    """
    # Detrend the series
    detrended = signal.detrend(price_series)
    
    # Compute FFT
    n = len(price_series)
    yf = fft(detrended)
    xf = fftfreq(n, sampling_period)
    
    # Get positive frequencies only
    pos_mask = xf > 0
    frequencies = xf[pos_mask]
    amplitudes = 2.0/n * np.abs(yf[pos_mask])
    
    # Find peaks
    peaks, _ = signal.find_peaks(amplitudes, height=np.mean(amplitudes))
    
    # Convert to periods and strengths
    cycles = {}
    for peak in peaks:
        period = 1/frequencies[peak]
        strength = amplitudes[peak]
        if 2 <= period <= 365:  # Only include cycles between 2 days and 1 year
            cycles[period] = strength
    
    return dict(sorted(cycles.items(), key=lambda x: x[1], reverse=True)[:5])

def identify_support_resistance(prices, window=21):
    """
    Identify support and resistance levels using kernel density estimation
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
        
    # Remove NaN values
    prices = prices[~np.isnan(prices)]
    
    if len(prices) == 0:
        return []
        
    kde = stats.gaussian_kde(prices)
    price_range = np.linspace(min(prices), max(prices), 100)
    density = kde(price_range)
    
    # Find peaks in density
    peaks, _ = signal.find_peaks(density)
    
    return price_range[peaks]

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced technical features
    """
    df = df.copy()
    
    # Basic returns and volatility
    df['returns'] = df['price_usd'].pct_change()
    df['log_returns'] = np.log(df['price_usd']).diff()
    
    # Volatility features
    df['realized_volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
    
    # Volume-based features
    df['dollar_volume'] = df['price_usd'] * df['volume_usd']
    df['relative_volume'] = df['volume_usd'] / df['volume_usd'].expanding().mean()
    
    # Multi-horizon momentum
    for window in [5, 21, 63]:
        # Price momentum
        df[f'momentum_{window}d'] = df['price_usd'].pct_change(window)
        
        # Volume momentum
        df[f'volume_momentum_{window}d'] = df['volume_usd'].pct_change(window)
        
        # Volatility regime
        df[f'volatility_regime_{window}d'] = (
            df['realized_volatility'] > 
            df['realized_volatility'].rolling(window).mean()
        ).astype(int)
        
        # Money flow strength
        df[f'money_flow_{window}d'] = (
            df['dollar_volume'].pct_change(window)
        )
    
    return df

def apply_expanding_scale(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Apply expanding window normalization to avoid look-ahead bias
    """
    df_scaled = df.copy()
    
    for feature in features:
        expanding_mean = df[feature].expanding().mean()
        expanding_std = df[feature].expanding().std()
        df_scaled[feature] = (df[feature] - expanding_mean) / expanding_std
        
    return df_scaled

def calculate_position_sizes(
    predictions: np.ndarray,
    volatility: np.ndarray,
    max_position: float = 0.1
) -> np.ndarray:
    """
    Calculate position sizes using volatility-adjusted Kelly criterion
    """
    confidence = abs(predictions)
    vol_scalar = 1 / volatility
    raw_positions = confidence * vol_scalar
    return np.clip(raw_positions, -max_position, max_position)

def calculate_portfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics
    """
    annualized_return = returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
    
    return {
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def calculate_phase_relationships(returns_dict):
    """
    Analyze phase relationships between different cryptocurrencies
    """
    phase_relationships = {}
    currencies = list(returns_dict.keys())
    
    for i, curr1 in enumerate(currencies):
        returns1 = returns_dict[curr1]
        if isinstance(returns1, pd.Series):
            returns1 = returns1.values
            
        for curr2 in currencies[i+1:]:
            returns2 = returns_dict[curr2]
            if isinstance(returns2, pd.Series):
                returns2 = returns2.values
                
            # Ensure no NaN values
            mask = ~(np.isnan(returns1) | np.isnan(returns2))
            if not np.any(mask):
                continue
                
            returns1_clean = returns1[mask]
            returns2_clean = returns2[mask]
            
            if len(returns1_clean) == 0 or len(returns2_clean) == 0:
                continue
            
            # Calculate cross-correlation
            xcorr = signal.correlate(returns1_clean, returns2_clean)
            lags = signal.correlation_lags(len(returns1_clean), len(returns2_clean))
            lag = lags[np.argmax(xcorr)]
            
            # Store relationship
            phase_relationships[f"{curr1}_vs_{curr2}"] = {
                'lag': int(lag),
                'correlation': float(np.max(xcorr))
            }
    
    return phase_relationships

def generate_cycle_forecast(df, dominant_cycles):
    """
    Generate price forecasts based on detected cycles
    """
    if isinstance(df, pd.DataFrame):
        prices = df['price_usd'].values
    else:
        prices = df
        
    forecast = np.zeros(30)  # 30-day forecast
    
    for period, strength in dominant_cycles.items():
        # Create sine wave for each cycle
        t = np.arange(30)
        cycle = strength * np.sin(2 * np.pi * t / period)
        forecast += cycle
    
    # Add trend
    trend = np.polyfit(np.arange(len(prices)), prices, 1)
    forecast_trend = np.poly1d(trend)(np.arange(len(prices), len(prices) + 30))
    
    return forecast + forecast_trend