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
    Enhanced cycle detection with multiple timeframe analysis
    """
    def get_timeframe_cycles(data, period):
        # Detrend the series
        detrended = signal.detrend(data)
        
        # Compute FFT
        n = len(data)
        yf = fft(detrended)
        xf = fftfreq(n, period)
        
        # Get positive frequencies
        pos_mask = xf > 0
        frequencies = xf[pos_mask]
        amplitudes = 2.0/n * np.abs(yf[pos_mask])
        
        # Find peaks with dynamic threshold
        threshold = np.mean(amplitudes) + np.std(amplitudes)
        peaks, _ = signal.find_peaks(amplitudes, height=threshold)
        
        return [(1/frequencies[peak], amplitudes[peak]) for peak in peaks]

    # Analyze multiple timeframes
    timeframes = {
        'short': price_series[-30:],    # Last month
        'medium': price_series[-90:],   # Last quarter
        'long': price_series[-365:],    # Last year
    }
    
    # Get cycles for each timeframe
    all_cycles = {}
    for timeframe_name, data in timeframes.items():
        if len(data) > 10:  # Minimum data requirement
            cycles = get_timeframe_cycles(data, sampling_period)
            for period, strength in cycles:
                if 2 <= period <= 365:  # Filter valid periods
                    if period not in all_cycles:
                        all_cycles[period] = {
                            'strength': strength,
                            'timeframes': [timeframe_name]
                        }
                    else:
                        all_cycles[period]['strength'] += strength
                        all_cycles[period]['timeframes'].append(timeframe_name)

    # Score cycles based on presence across timeframes
    final_cycles = {}
    for period, info in all_cycles.items():
        # Boost strength if cycle appears in multiple timeframes
        timeframe_boost = len(info['timeframes']) / len(timeframes)
        final_cycles[period] = info['strength'] * (1 + timeframe_boost)

    return dict(sorted(final_cycles.items(), key=lambda x: x[1], reverse=True)[:5])

def detect_market_regime(price_data, volume_data, window=30):
    """
    Detect market regime using multiple indicators
    """
    def calculate_hurst_exponent(prices, lags=range(2, 100)):
        # Calculate Hurst exponent to determine trend strength
        tau = []
        lagvec = []
        
        for lag in lags:
            prices_lag = np.log(prices[lag:]) - np.log(prices[:-lag])
            m = np.mean(np.abs(prices_lag))
            v = np.std(prices_lag)
            tau.append([np.log(lag), np.log(m)])
            lagvec.append(v)
            
        tau = np.array(tau)
        reg = np.polyfit(tau[:, 0], tau[:, 1], 1)
        return reg[0]  # Hurst exponent
    
    # Convert numpy arrays to pandas Series for diff operation
    price_series = pd.Series(price_data)
    volume_series = pd.Series(volume_data)
    
    # Calculate various regime indicators
    returns = price_series.pct_change().values[1:]  # Changed from diff
    log_returns = np.log(price_series).diff().values[1:]
    
    # Volatility regime
    rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
    vol_regime = rolling_vol > rolling_vol.rolling(window*2).mean()
    
    # Trend regime using Hurst exponent
    hurst = calculate_hurst_exponent(price_data[-min(len(price_data), 1000):])
    trend_strength = (hurst - 0.5) * 2  # Scale to -1 to 1
    
    # Volume regime
    volume_ma = volume_series.rolling(window).mean()
    volume_regime = volume_series > volume_ma
    
    # Momentum regime
    momentum = pd.Series(returns).rolling(window).mean()
    mom_regime = momentum > momentum.rolling(window*2).mean()
    
    # Combine indicators into regime score
    regime_scores = {
        'volatility': 1 if vol_regime.iloc[-1] else -1,
        'trend': np.sign(trend_strength),
        'volume': 1 if volume_regime.iloc[-1] else -1,
        'momentum': 1 if mom_regime.iloc[-1] else -1
    }
    
    # Calculate overall regime
    weights = {
        'volatility': 0.3,
        'trend': 0.3,
        'volume': 0.2,
        'momentum': 0.2
    }
    
    regime_score = sum(score * weights[key] for key, score in regime_scores.items())
    
    regimes = {
        'strong_bull': regime_score > 0.6,
        'bull': regime_score > 0.2,
        'neutral': abs(regime_score) <= 0.2,
        'bear': regime_score < -0.2,
        'strong_bear': regime_score < -0.6
    }
    
    current_regime = next(name for name, condition in regimes.items() if condition)
    
    return {
        'regime': current_regime,
        'score': regime_score,
        'indicators': regime_scores,
        'hurst': hurst
    }

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
def generate_cycle_forecast(df, dominant_cycles, market_data=None, confidence_threshold=0.3):
    """
    Generate price forecasts based on detected cycles with market regime awareness
    """
    if isinstance(df, pd.DataFrame):
        prices = df['price_usd'].values
        volumes = df['volume_usd'].values if 'volume_usd' in df else None
    else:
        prices = df
        volumes = None
    
    # Get market regime if not provided in market_data
    if market_data is None:
        market_data = {
            'volume_trend': 0,
            'market_breadth': 0.5
        }
    
    # Calculate prediction confidence with market data
    confidence = calculate_prediction_confidence(prices, dominant_cycles, market_data)
    
    # Only generate forecast if confidence is above threshold
    if confidence < confidence_threshold:
        return [], 0.0
        
    forecast = np.zeros(30)  # 30-day forecast
    
    # Add trend with market-based adjustment
    trend = np.polyfit(np.arange(len(prices)), prices, 1)
    trend_multiplier = 1.0
    
    # Adjust trend based on market breadth
    if market_data['market_breadth'] > 0.7:
        trend_multiplier = 1.2  # Strengthen trend in strong market
    elif market_data['market_breadth'] < 0.3:
        trend_multiplier = 0.8  # Weaken trend in weak market
    
    trend_dampening = np.exp(-0.05 * np.arange(30))
    forecast_trend = np.poly1d(trend)(np.arange(len(prices), len(prices) + 30))
    forecast_trend *= trend_dampening * trend_multiplier
    
    # Add cycles with market-aware adjustments
    for period, strength in dominant_cycles.items():
        t = np.arange(30)
        dampening = np.exp(-0.02 * t)
        
        # Adjust cycle strength based on market conditions
        if market_data['market_breadth'] > 0.7:
            strength *= 1.2  # Amplify cycles in strong market
        elif market_data['market_breadth'] < 0.3:
            strength *= 0.8  # Dampen cycles in weak market
        
        cycle = strength * np.sin(2 * np.pi * t / period) * dampening
        forecast += cycle
    
    final_forecast = forecast + forecast_trend
    
    # Scale forecast with market-aware bounds
    mean_price = np.mean(prices[-30:])
    
    # Adjust max deviation based on market conditions
    base_deviation = 0.3  # Base 30% deviation
    if market_data['market_breadth'] > 0.7:
        max_deviation = base_deviation * 1.2  # Allow larger moves in strong market
    elif market_data['market_breadth'] < 0.3:
        max_deviation = base_deviation * 0.8  # Restrict moves in weak market
    else:
        max_deviation = base_deviation
    
    # Volume trend can further modify bounds
    if market_data['volume_trend'] > 0:
        max_deviation *= 1.1  # More room for movement on rising volume
    
    forecast_bounds = np.clip(
        final_forecast, 
        mean_price * (1 - max_deviation),
        mean_price * (1 + max_deviation)
    )
    
    return forecast_bounds, confidence

# Improve confidence calculation
def calculate_prediction_confidence(price_data, cycles, market_data=None):
    """
    Enhanced confidence calculation incorporating market conditions
    """
    if not cycles:
        return 0.0
    
    # Basic cycle metrics
    cycle_strength = min(1.0, sum(cycles.values()) / (max(cycles.values()) * len(cycles)))
    cycle_consistency = len(cycles) / 5
    
    # Volatility analysis
    returns = np.diff(price_data) / price_data[:-1]
    volatility = np.std(returns) * np.sqrt(252)
    recent_volatility = np.std(returns[-30:]) * np.sqrt(252)
    vol_ratio = recent_volatility / volatility if volatility != 0 else 1
    
    # Market condition adjustments
    if market_data is not None:
        vol_trend = market_data.get('volume_trend', 0)
        vol_score = 1 + (0.2 * vol_trend)  # ±20% adjustment based on volume trend
        market_breadth = market_data.get('market_breadth', 0.5)
        breadth_score = 0.5 + (market_breadth / 2)  # Scale 0-1
    else:
        vol_score = 1
        breadth_score = 0.5

    # Combined confidence score with weights
    weights = {
        'cycle_strength': 0.3,
        'cycle_consistency': 0.2,
        'volatility': 0.2,
        'volume': 0.15,
        'market_breadth': 0.15
    }
    
    confidence = (
        weights['cycle_strength'] * cycle_strength +
        weights['cycle_consistency'] * cycle_consistency +
        weights['volatility'] * (1 / (1 + vol_ratio)) +
        weights['volume'] * vol_score +
        weights['market_breadth'] * breadth_score
    )
    
    return min(0.95, max(0.05, confidence))