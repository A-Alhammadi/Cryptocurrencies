import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict

from config import DB_CONFIG, CURRENCIES
from database import load_data_from_db
from analysis import (
    detect_cycles,
    identify_support_resistance,
    compute_advanced_features,
    apply_expanding_scale,
    calculate_position_sizes,
    calculate_portfolio_metrics,
    calculate_phase_relationships,
    generate_cycle_forecast
)

def process_single_currency(currency: str) -> Dict:
    """
    Process analysis for a single currency
    """
    try:
        # Load data
        df = load_data_from_db(DB_CONFIG, currency)
        if df.empty:
            return None
        
        # Compute features
        df = compute_advanced_features(df)
        
        # Define features for scaling
        features = [
            'realized_volatility',
            'relative_volume',
            'momentum_5d',
            'momentum_21d',
            'momentum_63d',
            'volume_momentum_5d',
            'volume_momentum_21d',
            'volume_momentum_63d',
            'volatility_regime_21d',
            'money_flow_21d'
        ]
        
        # Scale features
        df = apply_expanding_scale(df, features)
        
        # Detect cycles
        cycles = detect_cycles(df['price_usd'].values)
        
        # Identify support/resistance levels
        support_resistance = identify_support_resistance(df['price_usd'])
        
        # Generate forecast
        forecast = generate_cycle_forecast(df, cycles)
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics(df['returns'].dropna())
        
        return {
            'currency': currency,
            'cycles': cycles,
            'support_resistance': support_resistance,
            'forecast': forecast,
            'metrics': metrics,
            'data': df
        }
    except Exception as e:
        print(f"Error processing {currency}: {str(e)}")
        return None

def main():
    """
    Main function to run the complete analysis
    """
    print("Cryptocurrency Cycle Analysis")
    print("============================")
    
    # Process all currencies in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(CURRENCIES), 8)) as executor:
        futures = {executor.submit(process_single_currency, currency): currency 
                  for currency in CURRENCIES}
        
        for future in futures:
            currency = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results[currency] = result
                    print(f"✓ Successfully processed {currency}")
            except Exception as e:
                print(f"✗ Error processing {currency}: {str(e)}")
    
    # Calculate portfolio-level metrics
    portfolio_returns = pd.DataFrame({
        currency: results[currency]['data']['returns']
        for currency in results
    })
    
    portfolio_metrics = calculate_portfolio_metrics(portfolio_returns.mean(axis=1))
    
    # Calculate phase relationships
    phase_rels = calculate_phase_relationships(
        {curr: results[curr]['data']['returns'] for curr in results}
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'analysis_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("Crypto Analysis Results\n")
        f.write("======================\n\n")
        
        # Write portfolio results
        f.write("Portfolio Results\n")
        f.write("----------------\n")
        for metric, value in portfolio_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
        
        # Write individual currency results
        f.write("Individual Currency Results\n")
        f.write("-------------------------\n")
        for currency in results:
            f.write(f"\n{currency}:\n")
            
            # Metrics
            f.write("  Metrics:\n")
            for metric, value in results[currency]['metrics'].items():
                f.write(f"    {metric}: {value:.4f}\n")
            
            # Cycles
            f.write("\n  Dominant Cycles (days):\n")
            for period, strength in results[currency]['cycles'].items():
                f.write(f"    {period:.1f} days (strength: {strength:.2f})\n")
            
            # Support/Resistance
            f.write("\n  Support/Resistance Levels:\n")
            for level in results[currency]['support_resistance']:
                f.write(f"    ${level:.2f}\n")
            
            # Forecast
            forecast = results[currency]['forecast']
            direction = "Up" if forecast[-1] > forecast[0] else "Down"
            magnitude = abs(forecast[-1] - forecast[0])/forecast[0]
            f.write(f"\n  30-Day Forecast:\n")
            f.write(f"    Direction: {direction}\n")
            f.write(f"    Magnitude: {magnitude:.1%}\n")
            
        # Write phase relationships
        f.write("\nPhase Relationships\n")
        f.write("-----------------\n")
        for pair, stats in phase_rels.items():
            f.write(f"{pair}:\n")
            f.write(f"  Lag: {stats['lag']} days\n")
            f.write(f"  Correlation: {stats['correlation']:.2f}\n")
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()