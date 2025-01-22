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
# Fix in determine_optimal_window
def determine_optimal_window(df: pd.DataFrame, test_windows=[30, 60, 90]) -> Tuple[int, Dict]:
    """
    Determine optimal prediction window based on price characteristics
    """
    window_results = {}
    price_data = df['price_usd'].values
    
    for window in test_windows:
        # Skip if we don't have enough data
        if len(price_data) < window * 2:
            continue
            
        # Test predictions for this window
        predictions = []
        actuals = []
        
        for i in range(len(price_data) - window):
            train_data = price_data[i:i+window]
            actual = price_data[i+window]
            
            cycles = detect_cycles(train_data)
            if cycles:
                forecast, confidence = generate_cycle_forecast(train_data, cycles)  # Now properly unpacking tuple
                if len(forecast) > 0:
                    predictions.append(forecast[-1])
                    actuals.append(actual)
        
        if predictions and actuals:
            # Calculate metrics
            mape = np.mean(np.abs((np.array(predictions) - np.array(actuals)) / np.array(actuals)))
            window_results[window] = {
                'mape': mape,
                'n_predictions': len(predictions)
            }
    
    if not window_results:
        return test_windows[0], {'mape': float('inf'), 'n_predictions': 0}
    
    # Choose window with lowest MAPE
    optimal_window = min(window_results.items(), key=lambda x: x[1]['mape'])[0]
    return optimal_window, window_results[optimal_window]

# Improve the cycle forecast generation
def generate_cycle_forecast(df, dominant_cycles, confidence_threshold=0.3):
    """
    Generate price forecasts based on detected cycles with confidence score
    """
    if isinstance(df, pd.DataFrame):
        prices = df['price_usd'].values
    else:
        prices = df
        
    # Calculate prediction confidence
    confidence = calculate_prediction_confidence(prices, dominant_cycles)
    
    # Only generate forecast if confidence is above threshold
    if confidence < confidence_threshold:
        return [], 0.0
        
    forecast = np.zeros(30)  # 30-day forecast
    
    # Add trend first
    trend = np.polyfit(np.arange(len(prices)), prices, 1)
    trend_dampening = np.exp(-0.05 * np.arange(30))  # Gentle dampening of trend over time
    forecast_trend = np.poly1d(trend)(np.arange(len(prices), len(prices) + 30)) * trend_dampening
    
    # Add cycles with progressive dampening for longer forecasts
    for period, strength in dominant_cycles.items():
        t = np.arange(30)
        dampening = np.exp(-0.02 * t)  # Reduce cycle impact over time
        cycle = strength * np.sin(2 * np.pi * t / period) * dampening
        forecast += cycle
    
    final_forecast = forecast + forecast_trend
    
    # Scale forecast to avoid extreme values
    mean_price = np.mean(prices[-30:])  # Use recent mean for scaling
    max_deviation = 0.3  # Maximum 30% deviation from mean
    forecast_bounds = np.clip(
        final_forecast, 
        mean_price * (1 - max_deviation),
        mean_price * (1 + max_deviation)
    )
    
    return forecast_bounds, confidence

# Improve confidence calculation
def calculate_prediction_confidence(price_data, cycles, window_size=30):
    """
    Calculate prediction confidence with improved factors
    """
    if not cycles:
        return 0.0
    
    # Cycle strength and consistency
    cycle_strength = min(1.0, sum(cycles.values()) / (max(cycles.values()) * len(cycles)))
    cycle_consistency = len(cycles) / 5  # We look for max 5 cycles
    
    # Price stability
    returns = np.diff(price_data) / price_data[:-1]
    volatility = np.std(returns) * np.sqrt(252)
    stability = 1 / (1 + volatility)
    
    # Trend analysis
    trend_coef = np.polyfit(np.arange(len(price_data)), price_data, 1)[0]
    trend_strength = abs(trend_coef) / np.mean(price_data)
    trend_score = np.exp(-2 * trend_strength)  # Lower confidence for stronger trends
    
    # Recent performance
    recent_volatility = np.std(returns[-window_size:]) * np.sqrt(252)
    recent_stability = 1 / (1 + recent_volatility)
    
    # Combined confidence score with adjusted weights
    confidence = (0.3 * stability + 
                 0.2 * cycle_strength +
                 0.2 * cycle_consistency +
                 0.15 * trend_score +
                 0.15 * recent_stability)
    
    return min(0.95, max(0.05, confidence))  # Bound between 5% and 95%