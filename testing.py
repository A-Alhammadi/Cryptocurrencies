import pandas as pd
import numpy as np
from database import load_data_from_db
from analysis import *
from technical_analysis import get_technical_signals, backtest_signals
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

            # Add technical analysis signals
            df = get_technical_signals(df)
        
            # Run technical analysis backtest and get updated df
            df, technical_results = backtest_signals(df)
            
            predictions = []
            actuals = []
            accuracies = []
            dates = []
            confidences = []
            regimes = []      
            regime_scores = []
    
            
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
                    # Generate predictions
                    train_price = train_data['price_usd'].values

                    # Add price validation
                    if not np.all(np.isfinite(train_price)):
                        continue
                        
                    if np.std(train_price) < 1e-6:  # Skip periods with no price movement
                        continue

                    # Get market regime
                    regime_info = detect_market_regime(
                        train_price,
                        train_data['volume_usd'].values
                    )

                    cycles = detect_cycles(train_price)
                    if not cycles:
                        continue

                    # Generate forecast with market data
                    market_data = {
                        'volume_trend': np.sign(np.diff(train_data['volume_usd'].values[-30:]).mean()),
                        'market_breadth': (regime_info['score'] + 1) / 2
                    }

                    forecast, confidence = generate_cycle_forecast(train_data, cycles, market_data=market_data)
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
                    regimes.append(regime_info['regime'])
                    regime_scores.append(regime_info['score'])
                    
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
            regimes = [regimes[i] for i in range(len(valid_mask)) if valid_mask[i]]
            regime_scores = [regime_scores[i] for i in range(len(valid_mask)) if valid_mask[i]]
            
            if len(predictions) < min_predictions:
                return None
                
            mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            directional_accuracy = np.mean(accuracies) * 100

            # Calculate regime-specific accuracy
            regime_accuracy = {}
            for regime in set(regimes):
                regime_mask = np.array(regimes) == regime
                if any(regime_mask):
                    regime_preds = predictions[regime_mask]
                    regime_acts = actuals[regime_mask]
                    regime_accuracy[regime] = {
                        'count': sum(regime_mask),
                        'mape': np.mean(np.abs((regime_preds - regime_acts) / regime_acts)) * 100,
                        'accuracy': np.mean([accuracies[i] for i in range(len(accuracies)) 
                                        if regime_mask[i]]) * 100
                    }

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
                },
                'regimes': regimes,
                'regime_scores': regime_scores,
                'regime_accuracy': regime_accuracy,
                'technical_analysis': (df, technical_results),
                'data': df
            }
            
        except Exception as e:
            print(f"Error in validation for {currency}: {str(e)}")
            return None
    
    def plot_results(self, currency: str, results: Dict):
        """
        Plot validation results including technical analysis and save to results directory
        """
        if results is None or len(results['predictions']) < 2:
            print(f"Insufficient data to plot results for {currency}")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # First, create technical analysis plots
            if 'data' in results:
                df = results['data']
                
                # Create figure with subplots for technical analysis
                fig = plt.figure(figsize=(15, 20))
                gs = plt.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])
                
                # Price and predictions plot with EMAs
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])  # RSI
                ax3 = plt.subplot(gs[2])  # MACD
                ax4 = plt.subplot(gs[3])  # EMAs
                ax5 = plt.subplot(gs[4])  # Portfolio Value
                
                # Price plot with EMAs
                ax1.plot(df.index, df['price_usd'], label='Price', color='blue')
                ax1.plot(df.index, df['ema_50'], label='50 EMA', color='orange', alpha=0.7)
                ax1.plot(df.index, df['ema_200'], label='200 EMA', color='red', alpha=0.7)
                
                # Plot buy/sell signals
                buy_points = df[df['buy_signal']]['price_usd']
                sell_points = df[df['sell_signal']]['price_usd']
                ax1.scatter(buy_points.index, buy_points, color='green', marker='^', label='Buy Signal')
                ax1.scatter(sell_points.index, sell_points, color='red', marker='v', label='Sell Signal')
                
                # RSI plot
                ax2.plot(df.index, df['rsi'], color='purple')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax2.set_ylabel('RSI')
                ax2.grid(True)
                ax2.set_ylim(0, 100)
                
                # MACD plot
                ax3.plot(df.index, df['macd'], label='MACD', color='blue')
                ax3.plot(df.index, df['macd_signal'], label='Signal', color='orange')
                ax3.bar(df.index, df['macd_hist'], label='Histogram', color='gray', alpha=0.3)
                ax3.set_ylabel('MACD')
                ax3.legend()
                ax3.grid(True)
                
                # EMA Crossovers
                ax4.plot(df.index, df['ema_50'], label='50 EMA', color='orange')
                ax4.plot(df.index, df['ema_200'], label='200 EMA', color='red')
                ax4.fill_between(df.index, df['ema_50'], df['ema_200'],
                            where=df['ema_50'] >= df['ema_200'],
                            color='green', alpha=0.1)
                ax4.fill_between(df.index, df['ema_50'], df['ema_200'],
                            where=df['ema_50'] < df['ema_200'],
                            color='red', alpha=0.1)
                ax4.set_ylabel('EMAs')
                ax4.legend()
                ax4.grid(True)
                
                # Portfolio value plot
                ax5.plot(df.index, df['portfolio_value'], label='Portfolio Value', color='green')
                ax5.set_ylabel('Portfolio Value ($)')
                ax5.grid(True)
                
                # Formatting for all subplots
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                ax1.set_title(f'Technical Analysis - {currency}')
                ax1.legend()
                
                plt.tight_layout()
                tech_path = os.path.join(self.plots_dir, f'{currency}_technical_{timestamp}.png')
                plt.savefig(tech_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Now create the original cycle analysis plots
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
            
            # Create figure with subplots for cycle analysis
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
            
            # Price prediction plot with regime background
            if 'regimes' in results:
                regimes = np.array(results['regimes'])
                unique_regimes = np.unique(regimes)
                colors = {
                    'strong_bull': 'lightgreen',
                    'bull': 'palegreen',
                    'neutral': 'white',
                    'bear': 'salmon',
                    'strong_bear': 'lightcoral'
                }
                
                # Plot regime backgrounds
                for regime in unique_regimes:
                    mask = regimes == regime
                    if any(mask):
                        regime_dates = [dates[i] for i in range(len(dates)) if mask[i]]
                        ax1.axvspan(min(regime_dates), max(regime_dates), 
                                alpha=0.2, color=colors.get(regime, 'white'))
                                
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
            cycle_path = os.path.join(self.plots_dir, f'{currency}_backtest_{timestamp}.png')
            plt.savefig(cycle_path, dpi=300, bbox_inches='tight')
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
            
            print(f"Plots saved to: {tech_path}, {cycle_path}, and {confidence_path}")
            
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
Min Error: {np.min(np.abs((np.array(results['predictions']) - np.array(results['actuals'])) / np.array(results['actuals']) * 100)):.2f}%"""

            # Add Technical Analysis Results right after Price Statistics
            if 'technical_analysis' in results:
                # Handle the case where technical_analysis is a tuple (df, metrics)
                tech_metrics = results['technical_analysis'][1] if isinstance(results['technical_analysis'], tuple) else results['technical_analysis']
                
                report += f"""

Technical Analysis Results:
------------------------
Total Return: {tech_metrics['total_return']:.2%}
Sharpe Ratio: {tech_metrics['sharpe_ratio']:.2f}
Max Drawdown: {tech_metrics['max_drawdown']:.2%}
Number of Trades: {tech_metrics['number_of_trades']}
Win Rate: {tech_metrics['win_rate']:.2%}
Buy Signals: {tech_metrics['buy_signals']}
Sell Signals: {tech_metrics['sell_signals']}"""

            report += """

Market Regime Analysis:
--------------------"""

            if 'regime_accuracy' in results:
                for regime, stats in results['regime_accuracy'].items():
                    report += f"\n{regime.upper()}:\n"
                    report += f"  Count: {stats['count']}\n"
                    report += f"  MAPE: {stats['mape']:.2f}%\n"
                    report += f"  Directional Accuracy: {stats['accuracy']:.2f}%\n"

            report += "\nRecent Predictions:\n----------------\n"

            # Add last 5 predictions
            for i in range(min(5, len(results['predictions']))):
                idx = -(i+1)
                report += f"Date: {results['dates'][idx]} "
                report += f"Predicted: ${results['predictions'][idx]:.2f} "
                report += f"Actual: ${results['actuals'][idx]:.2f} "
                report += f"Confidence: {results['confidences'][idx]:.2%}"
                if 'regimes' in results:
                    report += f" Regime: {results['regimes'][idx].upper()}"
                report += "\n"
                    
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
                        'window_metrics': res['window_metrics'],
                        'regimes': res['regimes'],
                        'regime_scores': [float(x) for x in res['regime_scores']],
                        'regime_accuracy': res['regime_accuracy']
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