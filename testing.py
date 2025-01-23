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
            
            # Data validation
            if len(df) < training_window:
                print(f"Insufficient data for {currency}. Need at least {training_window} days.")
                return None

            # Remove any NaN values
            df = df.dropna(subset=['price_usd', 'volume_usd'])
            
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
            
            # Walk forward analysis with smaller steps
            step_size = min(optimal_window // 3, 7)  # Smaller steps for more test points
            min_predictions = 30  # Minimum number of predictions required
            
            for i in range(training_window, len(df)-optimal_window, step_size):
                try:
                    # Training data
                    train_data = df.iloc[i-training_window:i].copy()
                    
                    # Test data
                    test_data = df.iloc[i:i+optimal_window].copy()
                    
                    if len(train_data) < training_window * 0.9:  # Allow for some missing data
                        continue
                    
                    # Generate predictions
                    train_price = train_data['price_usd'].values
                    
                    # Add price validation
                    if not np.all(np.isfinite(train_price)):
                        continue
                        
                    if np.std(train_price) < 1e-6:  # Skip periods with no price movement
                        continue
                    
                    cycles = detect_cycles(train_price)
                    if not cycles:
                        continue
                        
                    forecast, confidence = generate_cycle_forecast(train_data, cycles)
                    if len(forecast) == 0 or not np.all(np.isfinite(forecast)):
                        continue
                    
                    current_price = train_data['price_usd'].iloc[-1]
                    future_price = test_data['price_usd'].iloc[-1]
                    predicted_price = forecast[-1]
                    
                    # Validate prices
                    if not all(np.isfinite([current_price, future_price, predicted_price])):
                        continue
                        
                    if abs((predicted_price - current_price) / current_price) > 0.5:
                        continue  # Skip unrealistic predictions (>50% change)
                    
                    # Record results
                    predictions.append(predicted_price)
                    actuals.append(future_price)
                    dates.append(test_data.index[-1])
                    confidences.append(confidence)
                    
                    # Calculate directional accuracy with minimum change threshold
                    min_change_threshold = 0.001  # 0.1% minimum change to count
                    pred_change = (predicted_price - current_price) / current_price
                    actual_change = (future_price - current_price) / current_price
                    
                    if abs(actual_change) < min_change_threshold:
                        continue  # Skip periods with minimal price change
                        
                    accuracies.append(np.sign(pred_change) == np.sign(actual_change))
                    
                except Exception as e:
                    print(f"Error in prediction iteration: {str(e)}")
                    continue
            
            # Validate minimum number of predictions
            if len(predictions) < min_predictions:
                print(f"Insufficient predictions for {currency}: {len(predictions)} < {min_predictions}")
                return None
                
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
            
            if len(predictions) < min_predictions:
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
        if results is None or len(results['predictions']) < 2:
            print(f"Insufficient data to plot results for {currency}")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Convert dates from strings back to datetime for plotting
            dates = [pd.to_datetime(d) for d in results['dates']]
            
            # Add min/max validation for plots
            predictions = np.array(results['predictions'])
            actuals = np.array(results['actuals'])
            
            # Remove extreme outliers
            valid_mask = np.abs(predictions - np.median(predictions)) < 5 * np.std(predictions)
            predictions = predictions[valid_mask]
            actuals = actuals[valid_mask]
            dates = [dates[i] for i in range(len(valid_mask)) if valid_mask[i]]
            confidences = np.array(results['confidences'])[valid_mask]
            
            if len(predictions) < 2:
                print(f"Insufficient valid data points after filtering for {currency}")
                return
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
            
            # Price prediction plot
            ax1.plot(dates, actuals, label='Actual', color='blue', linewidth=2)
            ax1.plot(dates, predictions, label='Predicted', color='red', linestyle='--', linewidth=2)
            
            # Add confidence bands
            ax1.fill_between(dates, 
                           predictions * (1 - 0.5 * (1 - confidences)),
                           predictions * (1 + 0.5 * (1 - confidences)),
                           color='red', alpha=0.2)
            
            ax1.set_title(f'Cycle Predictions vs Actuals - {currency}')
            ax1.legend()
            ax1.grid(True)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Add price labels on y-axis
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
            
            # Prediction error plot
            errors = np.array([(p - a) / a * 100 for p, a in zip(predictions, actuals)])
            ax2.hist(errors, bins=30, color='blue', alpha=0.6)
            ax2.axvline(x=0, color='red', linestyle='--')
            ax2.set_title('Prediction Error Distribution (%)')
            ax2.set_xlabel('Prediction Error %')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
            
            # Scatter plot: Error vs Confidence
            ax3.scatter(dates, errors, c=confidences, cmap='viridis', 
                       alpha=0.6, s=50)
            ax3.set_title('Prediction Errors Over Time (colored by confidence)')
            ax3.set_ylabel('Prediction Error %')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            ax3.grid(True)
            
            # Add colorbar
            sc = ax3.scatter(dates, errors, c=confidences, cmap='viridis')
            plt.colorbar(sc, ax=ax3, label='Confidence Score')
            
            # Adjust layout and save
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, f'{currency}_backtest_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create additional plot for confidence analysis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Confidence over time
            ax1.plot(dates, confidences, color='green', linewidth=2)
            ax1.set_title(f'Prediction Confidence Over Time - {currency}')
            ax1.set_ylabel('Confidence Score')
            ax1.grid(True)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Accuracy vs Confidence scatter
            abs_errors = np.abs(errors)
            ax2.scatter(confidences, abs_errors, alpha=0.6)
            ax2.set_title('Prediction Accuracy vs Confidence')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Absolute Error %')
            ax2.grid(True)
            
            # Add trend line
            z = np.polyfit(confidences, abs_errors, 1)
            p = np.poly1d(z)
            ax2.plot(confidences, p(confidences), "r--", alpha=0.8, 
                    label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            ax2.legend()
            
            # Save confidence analysis plot
            plt.tight_layout()
            confidence_path = os.path.join(self.plots_dir, f'{currency}_confidence_analysis_{timestamp}.png')
            plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved to: {save_path} and {confidence_path}")
            
        except Exception as e:
            print(f"Error plotting results for {currency}: {str(e)}")
            plt.close('all')
    
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
    
    # Updated test period: 2023-2025
    start_date = '2023-01-01'
    end_date = '2025-01-01'
    
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
        f.write(f"Test Period: {start_date} to {end_date}\n\n")
        
        for currency in CURRENCIES:
            print(f"\nTesting {currency}...")
            try:
                results = backtest.validate_cycle_prediction(
                    currency, 
                    start_date, 
                    end_date
                )
                
                if results is not None:
                    if len(results['predictions']) >= 30:  # Minimum predictions threshold
                        successful_tests += 1
                        total_mape += results['mape']
                        total_accuracy += results['directional_accuracy']
                        total_confidence += np.mean(results['confidences'])
                        all_results[currency] = results
                        print(f"✓ Successfully processed {currency} with {len(results['predictions'])} predictions")
                    else:
                        print(f"✗ Insufficient predictions for {currency}: {len(results['predictions'])} < 30")
                else:
                    print(f"✗ No valid results for {currency}")
                
                # Write results to file
                report = backtest.generate_summary_report(currency, results)
                f.write(report)
                f.write("\n" + "="*50 + "\n")
                
                # Plot results if we have enough valid predictions
                if results is not None and len(results['predictions']) >= 30:
                    backtest.plot_results(currency, results)
                
            except Exception as e:
                print(f"Error testing {currency}: {str(e)}")
                f.write(f"\nError testing {currency}: {str(e)}\n")
        
        # Write summary statistics
        if successful_tests > 0:
            summary = f"""
Overall Summary
==============
Test Period: {start_date} to {end_date}
Successfully tested currencies: {successful_tests}
Average MAPE: {total_mape/successful_tests:.2f}%
Average Directional Accuracy: {total_accuracy/successful_tests:.2f}%
Average Confidence Score: {total_confidence/successful_tests:.2%}
"""
            f.write(summary)
            print(summary)
        else:
            summary = "\nNo successful tests completed.\n"
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