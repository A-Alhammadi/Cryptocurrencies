import pandas as pd
import numpy as np
from database import load_data_from_db
from analysis import *
from config import DB_CONFIG, CURRENCIES
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

class CycleAnalysisBacktest:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        
        # Create results directory if it doesn't exist
        self.results_dir = 'results'
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def validate_cycle_prediction(self, 
                                currency: str,
                                start_date: str,
                                end_date: str,
                                test_windows=[30, 60, 90],
                                training_window: int = 180) -> Dict:
        """
        Validate cycle predictions against historical data with adaptive windows
        """
        print(f"\nValidating predictions for {currency}")
        
        try:
            # Load data
            print(f"\nAttempting to load data for {currency}...")
            df = load_data_from_db(
                self.db_config, 
                currency, 
                start_date, 
                end_date
            )
            
            if df.empty:
                print(f"No data available for {currency}")
                return None
                
            print(f"Data loaded successfully:")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Price range: ${df['price_usd'].min():.2f} to ${df['price_usd'].max():.2f}")
            
            # Determine optimal prediction window
            optimal_window, window_metrics = determine_optimal_window(df, test_windows)
            print(f"\nOptimal prediction window for {currency}: {optimal_window} days")
            print(f"Window selection metrics: MAPE={window_metrics['mape']:.2%}, n_predictions={window_metrics['n_predictions']}")
            
            # Ensure we have enough data
            if len(df) < training_window + optimal_window:
                print(f"Insufficient data for {currency}. Need at least {training_window + optimal_window} days, got {len(df)} days.")
                return None
                
            # Compute features
            df = compute_advanced_features(df)
            
            predictions = []
            actuals = []
            accuracies = []
            dates = []
            confidences = []
            
            # Walk forward analysis
            step_size = optimal_window  # Move forward by optimal window size
            for i in range(training_window, len(df)-optimal_window, step_size):
                try:
                    # Training data
                    train_data = df.iloc[i-training_window:i].copy()
                    
                    # Test data
                    test_data = df.iloc[i:i+optimal_window].copy()
                    
                    if train_data.empty or test_data.empty:
                        continue
                    
                    # Generate predictions
                    train_price = train_data['price_usd'].values
                    if len(train_price) < training_window:
                        continue
                        
                    cycles = detect_cycles(train_price)
                    if not cycles:
                        continue
                        
                    forecast, confidence = generate_cycle_forecast(train_data, cycles)
                    if len(forecast) == 0:
                        continue
                    
                    current_price = train_data['price_usd'].iloc[-1]
                    future_price = test_data['price_usd'].iloc[-1]
                    predicted_price = forecast[-1]
                    
                    # Record results
                    predictions.append(predicted_price)
                    actuals.append(future_price)
                    dates.append(test_data.index[-1])  # Using index since we set date as index
                    confidences.append(confidence)
                    
                    # Calculate directional accuracy
                    pred_direction = np.sign(predicted_price - current_price)
                    actual_direction = np.sign(future_price - current_price)
                    accuracies.append(pred_direction == actual_direction)
                    
                    print(f"Generated prediction for {currency} at {test_data.index[-1]}:")
                    print(f"  Predicted: ${predicted_price:.2f}")
                    print(f"  Actual: ${future_price:.2f}")
                    print(f"  Confidence: {confidence:.2%}")
                    
                except Exception as e:
                    print(f"Error in prediction iteration: {str(e)}")
                    continue
            
            if not predictions or not actuals:
                print(f"No valid predictions generated for {currency}")
                return None
                
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            confidences = np.array(confidences)
            
            # Remove any remaining infinities or NaNs
            valid_mask = np.isfinite(predictions) & np.isfinite(actuals)
            predictions = predictions[valid_mask]
            actuals = actuals[valid_mask]
            confidences = confidences[valid_mask]
            dates = [dates[i] for i in range(len(valid_mask)) if valid_mask[i]]
            
            if len(predictions) == 0:
                return None
                
            mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            directional_accuracy = np.mean(accuracies) * 100
            
            return {
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'predictions': predictions.tolist(),
                'actuals': actuals.tolist(),
                'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates],
                'confidences': confidences.tolist(),
                'optimal_window': optimal_window,
                'window_metrics': {
                    'mape': float(window_metrics['mape']),
                    'n_predictions': int(window_metrics['n_predictions'])
                }
            }
            
        except Exception as e:
            print(f"Error in validation for {currency}: {str(e)}")
            return None
    
    def plot_results(self, currency: str, results: Dict):
        """
        Plot validation results and save to results directory
        """
        if results is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.figure(figsize=(15, 12))
        
        try:
            # Price prediction plot
            plt.subplot(3, 1, 1)
            plt.plot(range(len(results['actuals'])), results['actuals'], label='Actual', color='blue')
            plt.plot(range(len(results['predictions'])), results['predictions'], label='Predicted', color='red', linestyle='--')
            plt.title(f'Cycle Predictions vs Actuals - {currency}')
            plt.legend()
            
            # Prediction error plot
            plt.subplot(3, 1, 2)
            errors = np.array([(p - a) / a * 100 for p, a in zip(results['predictions'], results['actuals'])])
            plt.hist(errors, bins=50)
            plt.title('Prediction Error Distribution (%)')
            plt.xlabel('Prediction Error %')
            plt.ylabel('Frequency')
            
            # Confidence vs Error plot
            plt.subplot(3, 1, 3)
            plt.scatter(results['confidences'], abs(errors), alpha=0.5)
            plt.title('Prediction Confidence vs Error')
            plt.xlabel('Confidence Score')
            plt.ylabel('Absolute Error %')
            
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, f'{currency}_backtest_{timestamp}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Plot saved to: {save_path}")
            
        except Exception as e:
            print(f"Error plotting results for {currency}: {str(e)}")
            plt.close()
    
    def generate_summary_report(self, currency: str, results: Dict) -> str:
        """
        Generate a summary report of the backtesting results
        """
        if results is None:
            return f"No valid results available for {currency}\n"
            
        try:
            report = f"""
Cycle Analysis Backtest Results for {currency}
============================================

Prediction Settings:
------------------
Optimal Window: {results['optimal_window']} days
Window Selection MAPE: {results['window_metrics']['mape']:.2%}
Number of Test Predictions: {results['window_metrics']['n_predictions']}

Prediction Accuracy:
------------------
MAPE: {results['mape']:.2f}%
Directional Accuracy: {results['directional_accuracy']:.2f}%
Average Confidence Score: {np.mean(results['confidences']):.2%}

Price Statistics:
---------------
Average Actual Price: ${np.mean(results['actuals']):.2f}
Average Predicted Price: ${np.mean(results['predictions']):.2f}
Max Error: {np.max(np.abs((np.array(results['predictions']) - np.array(results['actuals'])) / np.array(results['actuals']) * 100)):.2f}%
Min Error: {np.min(np.abs((np.array(results['predictions']) - np.array(results['actuals'])) / np.array(results['actuals']) * 100)):.2f}%

Recent Predictions:
----------------
"""
            # Add last 5 predictions
            for i in range(min(5, len(results['predictions']))):
                idx = -(i+1)
                report += f"Date: {results['dates'][idx]} "
                report += f"Predicted: ${results['predictions'][idx]:.2f} "
                report += f"Actual: ${results['actuals'][idx]:.2f} "
                report += f"Confidence: {results['confidences'][idx]:.2%}\n"
                
            return report
            
        except Exception as e:
            print(f"Error generating report for {currency}: {str(e)}")
            return f"Error generating report for {currency}: {str(e)}\n"

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
    results_file = os.path.join(backtest.results_dir, f'backtest_results_{timestamp}.txt')
    json_file = os.path.join(backtest.results_dir, f'backtest_data_{timestamp}.json')
    
    all_results = {}
    successful_tests = 0
    total_mape = 0
    total_accuracy = 0
    total_confidence = 0
    
    print(f"\nResults will be saved to:")
    print(f"Text report: {results_file}")
    print(f"JSON data: {json_file}")
    print(f"Plots directory: {backtest.plots_dir}")
    
    with open(results_file, 'w') as f:
        f.write("Cryptocurrency Cycle Analysis Backtest Results\n")
        f.write("==========================================\n\n")
        
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
                    total_confidence += np.mean(results['confidences'])
                    all_results[currency] = results
                
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
            summary = f"""
Overall Summary
==============
Successfully tested currencies: {successful_tests}
Average MAPE: {total_mape/successful_tests:.2f}%
Average Directional Accuracy: {total_accuracy/successful_tests:.2f}%
Average Confidence Score: {total_confidence/successful_tests:.2%}
"""
            f.write(summary)
            print(summary)
    
    try:
        # Save detailed results to JSON for later analysis
        with open(json_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for curr, res in all_results.items():
                if res is not None:
                    json_results[curr] = {
                        'mape': float(res['mape']),
                        'directional_accuracy': float(res['directional_accuracy']),
                        'predictions': [float(x) for x in res['predictions']],
                        'actuals': [float(x) for x in res['actuals']],
                        'dates': res['dates'],
                        'confidences': [float(x) for x in res['confidences']],
                        'optimal_window': int(res['optimal_window']),
                        'window_metrics': res['window_metrics']
                    }
            json.dump(json_results, f, indent=4)
        
        print(f"\nBacktest completed successfully!")
        print(f"Text report saved to: {results_file}")
        print(f"JSON data saved to: {json_file}")
        print(f"Plots saved in: {backtest.plots_dir}")
        
    except Exception as e:
        print(f"Error saving JSON results: {str(e)}")

if __name__ == "__main__":
    run_full_backtest()