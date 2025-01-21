import pandas as pd
import numpy as np
from database import load_data_from_db
from analysis import *
from config import DB_CONFIG, CURRENCIES
import matplotlib.pyplot as plt
from datetime import datetime

class CycleAnalysisBacktest:
    def __init__(self, db_config: dict):
        self.db_config = db_config
            
    def validate_cycle_prediction(self, 
                                currency: str,
                                start_date: str,
                                end_date: str,
                                prediction_window: int = 30,
                                training_window: int = 180) -> Dict:  # Reduced from 365 to 180 days
        """
        Validate cycle predictions against historical data
        """
        print(f"\nValidating predictions for {currency}")
        
        # Load data
        print(f"\nAttempting to load data for {currency}...")
        df = load_data_from_db(
            self.db_config, 
            currency, 
            start_date, 
            end_date
        )
        print(f"Rows loaded: {len(df) if not df.empty else 0}")
        if not df.empty:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Price range: ${df['price_usd'].min():.2f} to ${df['price_usd'].max():.2f}")
        
        if df.empty:
            print(f"No data available for {currency}")
            return None
            
        # Ensure we have enough data
        if len(df) < training_window + prediction_window:
            print(f"Insufficient data for {currency}. Need at least {training_window + prediction_window} days, got {len(df)} days.")
            return None
            
        # Compute features
        df = compute_advanced_features(df)
        
        predictions = []
        actuals = []
        accuracies = []
        dates = []
        
        # Walk forward analysis
        step_size = prediction_window  # Move forward by prediction window size
        for i in range(training_window, len(df)-prediction_window, step_size):
            try:
                # Training data
                train_data = df.iloc[i-training_window:i].copy()
                
                # Test data
                test_data = df.iloc[i:i+prediction_window].copy()
                
                if train_data.empty or test_data.empty:
                    continue
                
                # Generate predictions
                train_price = train_data['price_usd'].values
                if len(train_price) < training_window:
                    continue
                    
                cycles = detect_cycles(train_price)
                if not cycles:
                    continue
                    
                forecast = generate_cycle_forecast(train_data, cycles)
                if len(forecast) == 0:
                    continue
                
                current_price = train_data['price_usd'].iloc[-1]
                future_price = test_data['price_usd'].iloc[-1]
                predicted_price = forecast[-1]  # Direct forecast rather than relative change
                
                # Record results
                predictions.append(predicted_price)
                actuals.append(future_price)
                dates.append(test_data.index[-1])
                
                # Calculate directional accuracy
                pred_direction = np.sign(predicted_price - current_price)
                actual_direction = np.sign(future_price - current_price)
                accuracies.append(pred_direction == actual_direction)
                
                print(f"Generated prediction for {currency} at {test_data.index[-1]}: Predicted ${predicted_price:.2f}, Actual ${future_price:.2f}")
                
            except Exception as e:
                print(f"Error in iteration: {str(e)}")
                continue
        
        if not predictions or not actuals:
            print(f"No valid predictions generated for {currency}")
            return None
            
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Remove any remaining infinities or NaNs
        valid_mask = np.isfinite(predictions) & np.isfinite(actuals)
        predictions = predictions[valid_mask]
        actuals = actuals[valid_mask]
        
        if len(predictions) == 0:
            return None
            
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        directional_accuracy = np.mean(accuracies) * 100
        
        return {
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates
        }
        
        if not predictions or not actuals:
            print(f"No valid predictions generated for {currency}")
            return None
            
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Remove any remaining infinities or NaNs
        valid_mask = np.isfinite(predictions) & np.isfinite(actuals)
        predictions = predictions[valid_mask]
        actuals = actuals[valid_mask]
        
        if len(predictions) == 0:
            return None
            
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        directional_accuracy = np.mean(accuracies) * 100
        
        return {
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def plot_results(self, currency: str, results: Dict):
        """
        Plot validation results
        """
        if results is None:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Price prediction plot
        plt.subplot(2, 1, 1)
        plt.plot(results['actuals'], label='Actual', color='blue')
        plt.plot(results['predictions'], label='Predicted', color='red', linestyle='--')
        plt.title(f'Cycle Predictions vs Actuals - {currency}')
        plt.legend()
        
        # Prediction error plot
        plt.subplot(2, 1, 2)
        errors = (results['predictions'] - results['actuals']) / results['actuals'] * 100
        plt.hist(errors, bins=50)
        plt.title('Prediction Error Distribution (%)')
        plt.xlabel('Prediction Error %')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'backtest_{currency}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def generate_summary_report(self, currency: str, results: Dict) -> str:
        """
        Generate a summary report of the backtesting results
        """
        if results is None:
            return f"No valid results available for {currency}\n"
            
        report = f"""
Cycle Analysis Backtest Results for {currency}
============================================

Prediction Accuracy:
------------------
MAPE: {results['mape']:.2f}%
Directional Accuracy: {results['directional_accuracy']:.2f}%

Additional Statistics:
-------------------
Number of Predictions: {len(results['predictions'])}
Average Actual Price: ${np.mean(results['actuals']):.2f}
Average Predicted Price: ${np.mean(results['predictions']):.2f}
Max Error: {np.max(np.abs((results['predictions'] - results['actuals']) / results['actuals'] * 100)):.2f}%
"""
        return report

def run_full_backtest():
    """
    Run complete backtest across all currencies
    """
    print("Starting Cryptocurrency Cycle Analysis Backtest")
    print("============================================")
    
    backtest = CycleAnalysisBacktest(DB_CONFIG)
    
    # Test period: 2022-2023
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'backtest_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("Cryptocurrency Cycle Analysis Backtest Results\n")
        f.write("==========================================\n\n")
        
        successful_tests = 0
        total_mape = 0
        total_accuracy = 0
        
        for currency in CURRENCIES:
            print(f"\nTesting {currency}...")
            try:
                results = backtest.validate_cycle_prediction(
                    currency, 
                    start_date, 
                    end_date
                )
                
                if results is not None:
                    successful_tests += 1
                    total_mape += results['mape']
                    total_accuracy += results['directional_accuracy']
                
                # Write results to file
                report = backtest.generate_summary_report(currency, results)
                f.write(report)
                f.write("\n" + "="*50 + "\n")
                
                # Plot results
                if results is not None:
                    backtest.plot_results(currency, results)
                
            except Exception as e:
                print(f"Error testing {currency}: {str(e)}")
                f.write(f"\nError testing {currency}: {str(e)}\n")
        
        # Write summary statistics
        if successful_tests > 0:
            f.write("\nOverall Summary\n")
            f.write("==============\n")
            f.write(f"Successfully tested currencies: {successful_tests}\n")
            f.write(f"Average MAPE: {total_mape/successful_tests:.2f}%\n")
            f.write(f"Average Directional Accuracy: {total_accuracy/successful_tests:.2f}%\n")
    
    print(f"\nBacktest results saved to {results_file}")

if __name__ == "__main__":
    run_full_backtest()