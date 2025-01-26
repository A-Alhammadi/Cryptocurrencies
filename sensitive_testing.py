import pandas as pd
import numpy as np
import psycopg2
import os
from datetime import datetime
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import DB_CONFIG, CURRENCIES

class SensitiveStrategyBacktest:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        
        # Create results directory if it doesn't exist
        self.results_dir = 'sensitive_results'
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_data_from_db(self, currency: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load cryptocurrency data from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            query = """
            SELECT 
                date::date as date,
                close_price as price_usd,
                volume as volume_usd,
                open_price,
                high_price,
                low_price
            FROM crypto_prices
            WHERE currency = %s
            AND date BETWEEN %s AND %s
            ORDER BY date ASC;
            """
            
            df = pd.read_sql(query, conn, params=[currency, start_date, end_date])
            conn.close()
            
            # Convert date column to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True, drop=False)
            
            return df
            
        except Exception as e:
            print(f"Error loading {currency}: {str(e)}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        df = df.copy()
        
        # Calculate RSI
        delta = df['price_usd'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['price_usd'].ewm(span=12, adjust=False).mean()
        exp2 = df['price_usd'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Short-term EMAs
        df['ema_10'] = df['price_usd'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['price_usd'].ewm(span=20, adjust=False).mean()
        
        # Stochastic Oscillator
        high_14 = df['price_usd'].rolling(window=14).max()
        low_14 = df['price_usd'].rolling(window=14).min()
        df['k_percent'] = ((df['price_usd'] - low_14) / (high_14 - low_14)) * 100
        df['d_percent'] = df['k_percent'].rolling(window=3).mean()
        
        # Volume-weighted RSI
        volume_multiplier = df['volume_usd'] / df['volume_usd'].rolling(window=14).mean()
        df['volume_weighted_rsi'] = df['rsi'] * volume_multiplier
        
        return df
    def generate_sensitive_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sensitive trading signals"""
        df = df.copy()
        
        # Combined sensitive signals
        df['sensitive_buy'] = (
            # Short-term EMA cross with RSI confirmation
            (df['ema_10'] > df['ema_20']) & (df['volume_weighted_rsi'] < 40) |
            
            # Stochastic oversold with positive MACD momentum
            (df['k_percent'] < 20) & (df['d_percent'] < 20) & (df['macd_hist'] > 0) |
            
            # Strong volume spike with oversold conditions
            (df['volume_weighted_rsi'] < 25) |
            
            # Multiple indicator confluence
            (df['k_percent'] > df['d_percent']) & (df['macd'] > df['macd_signal']) & (df['rsi'] < 45)
        )
        
        df['sensitive_sell'] = (
            # Short-term EMA cross with RSI confirmation
            (df['ema_10'] < df['ema_20']) & (df['volume_weighted_rsi'] > 60) |
            
            # Stochastic overbought with negative MACD momentum
            (df['k_percent'] > 80) & (df['d_percent'] > 80) & (df['macd_hist'] < 0) |
            
            # Strong volume spike with overbought conditions
            (df['volume_weighted_rsi'] > 75) |
            
            # Multiple indicator confluence
            (df['k_percent'] < df['d_percent']) & (df['macd'] < df['macd_signal']) & (df['rsi'] > 55)
        )
        
        return df

    def generate_individual_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate individual trading signals for each indicator"""
        df = df.copy()
        
        # 1. EMA Crossover Signals (10 & 20)
        df['ema_buy'] = (df['ema_10'] > df['ema_20']) & (df['ema_10'].shift(1) <= df['ema_20'].shift(1))
        df['ema_sell'] = (df['ema_10'] < df['ema_20']) & (df['ema_10'].shift(1) >= df['ema_20'].shift(1))
        
        # 2. RSI Signals
        df['rsi_buy'] = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
        df['rsi_sell'] = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
        
        # 3. MACD Signals
        df['macd_buy'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_sell'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # 4. Stochastic Signals
        df['stoch_buy'] = (df['k_percent'] < 20) & (df['k_percent'].shift(1) >= 20)
        df['stoch_sell'] = (df['k_percent'] > 80) & (df['k_percent'].shift(1) <= 80)
        
        # 5. Volume-weighted RSI Signals
        df['vol_rsi_buy'] = (df['volume_weighted_rsi'] < 30) & (df['volume_weighted_rsi'].shift(1) >= 30)
        df['vol_rsi_sell'] = (df['volume_weighted_rsi'] > 70) & (df['volume_weighted_rsi'].shift(1) <= 70)
        
        return df
    
    def backtest_individual_strategies(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
        """Run backtest for each individual signal type"""
        strategies = {
            'EMA_Cross': ('ema_buy', 'ema_sell'),
            'RSI': ('rsi_buy', 'rsi_sell'),
            'MACD': ('macd_buy', 'macd_sell'),
            'Stochastic': ('stoch_buy', 'stoch_sell'),
            'Volume_RSI': ('vol_rsi_buy', 'vol_rsi_sell')
        }
        
        results = {}
        
        for strategy_name, (buy_signal, sell_signal) in strategies.items():
            strategy_results = {
                'trades': [],
                'initial_capital': initial_capital,
                'final_capital': initial_capital,
                'total_trades': 0,
                'profitable_trades': 0,
                'buy_and_hold_return': 0,
                'strategy_return': 0,
                'price_history': []
            }
            
            # Calculate buy and hold return
            start_price = df['price_usd'].iloc[0]
            end_price = df['price_usd'].iloc[-1]
            strategy_results['buy_and_hold_return'] = ((end_price - start_price) / start_price) * 100
            
            # Trading simulation
            current_position = 'cash'
            current_capital = initial_capital
            entry_price = 0
            entry_date = None
            
            for idx, row in df.iterrows():
                strategy_results['price_history'].append({
                    'date': idx,
                    'price': row['price_usd'],
                    'capital': current_capital
                })
                
                if current_position == 'cash' and row[buy_signal]:
                    entry_price = row['price_usd']
                    entry_date = idx
                    current_position = 'invested'
                    
                elif current_position == 'invested' and row[sell_signal]:
                    exit_price = row['price_usd']
                    exit_date = idx
                    
                    trade_return = ((exit_price - entry_price) / entry_price) * 100
                    current_capital = current_capital * (1 + trade_return/100)
                    
                    trade = {
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return_pct': trade_return,
                        'capital_after_trade': current_capital,
                        'holding_period': (exit_date - entry_date).days
                    }
                    strategy_results['trades'].append(trade)
                    
                    strategy_results['total_trades'] += 1
                    if trade_return > 0:
                        strategy_results['profitable_trades'] += 1
                        
                    current_position = 'cash'
            
            # Close any open position at the end
            if current_position == 'invested':
                exit_price = df['price_usd'].iloc[-1]
                exit_date = df.index[-1]
                
                trade_return = ((exit_price - entry_price) / entry_price) * 100
                current_capital = current_capital * (1 + trade_return/100)
                
                trade = {
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return,
                    'capital_after_trade': current_capital,
                    'holding_period': (exit_date - entry_date).days
                }
                strategy_results['trades'].append(trade)
                strategy_results['total_trades'] += 1
                if trade_return > 0:
                    strategy_results['profitable_trades'] += 1
            
            # Calculate final metrics
            strategy_results['final_capital'] = current_capital
            strategy_results['strategy_return'] = ((current_capital - initial_capital) / initial_capital) * 100
            
            if strategy_results['total_trades'] > 0:
                strategy_results['win_rate'] = (strategy_results['profitable_trades'] / strategy_results['total_trades']) * 100
                trade_returns = [t['return_pct'] for t in strategy_results['trades']]
                holding_periods = [t['holding_period'] for t in strategy_results['trades']]
                
                strategy_results['avg_trade_return'] = np.mean(trade_returns)
                strategy_results['max_trade_return'] = np.max(trade_returns)
                strategy_results['min_trade_return'] = np.min(trade_returns)
                strategy_results['trade_return_std'] = np.std(trade_returns)
                strategy_results['avg_holding_period'] = np.mean(holding_periods)
                strategy_results['max_holding_period'] = np.max(holding_periods)
                strategy_results['min_holding_period'] = np.min(holding_periods)
            else:
                strategy_results['win_rate'] = 0
                strategy_results['avg_trade_return'] = 0
                strategy_results['max_trade_return'] = 0
                strategy_results['min_trade_return'] = 0
                strategy_results['trade_return_std'] = 0
                strategy_results['avg_holding_period'] = 0
                strategy_results['max_holding_period'] = 0
                strategy_results['min_holding_period'] = 0
            
            results[strategy_name] = strategy_results
            
        return results
    


    def backtest_strategy(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
        """Run backtest on the sensitive strategy"""
        results = {
            'trades': [],
            'initial_capital': initial_capital,
            'final_capital': initial_capital,
            'total_trades': 0,
            'profitable_trades': 0,
            'buy_and_hold_return': 0,
            'strategy_return': 0,
            'price_history': []
        }
        
        # Calculate buy and hold return
        start_price = df['price_usd'].iloc[0]
        end_price = df['price_usd'].iloc[-1]
        results['buy_and_hold_return'] = ((end_price - start_price) / start_price) * 100
        
        # Trading simulation
        current_position = 'cash'
        current_capital = initial_capital
        entry_price = 0
        entry_date = None
        
        for idx, row in df.iterrows():
            results['price_history'].append({
                'date': idx,
                'price': row['price_usd'],
                'capital': current_capital
            })
            
            if current_position == 'cash' and row['sensitive_buy']:
                entry_price = row['price_usd']
                entry_date = idx
                current_position = 'invested'
                
            elif current_position == 'invested' and row['sensitive_sell']:
                exit_price = row['price_usd']
                exit_date = idx
                
                trade_return = ((exit_price - entry_price) / entry_price) * 100
                current_capital = current_capital * (1 + trade_return/100)
                
                trade = {
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return,
                    'capital_after_trade': current_capital,
                    'holding_period': (exit_date - entry_date).days
                }
                results['trades'].append(trade)
                
                results['total_trades'] += 1
                if trade_return > 0:
                    results['profitable_trades'] += 1
                    
                current_position = 'cash'
        
        # If still invested at the end, close position
        if current_position == 'invested':
            exit_price = df['price_usd'].iloc[-1]
            exit_date = df.index[-1]
            
            trade_return = ((exit_price - entry_price) / entry_price) * 100
            current_capital = current_capital * (1 + trade_return/100)
            
            trade = {
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': trade_return,
                'capital_after_trade': current_capital,
                'holding_period': (exit_date - entry_date).days
            }
            results['trades'].append(trade)
            results['total_trades'] += 1
            if trade_return > 0:
                results['profitable_trades'] += 1
        
        # Calculate final metrics
        results['final_capital'] = current_capital
        results['strategy_return'] = ((current_capital - initial_capital) / initial_capital) * 100
        
        if results['total_trades'] > 0:
            results['win_rate'] = (results['profitable_trades'] / results['total_trades']) * 100
            trade_returns = [t['return_pct'] for t in results['trades']]
            holding_periods = [t['holding_period'] for t in results['trades']]
            
            results['avg_trade_return'] = np.mean(trade_returns)
            results['max_trade_return'] = np.max(trade_returns)
            results['min_trade_return'] = np.min(trade_returns)
            results['trade_return_std'] = np.std(trade_returns)
            results['avg_holding_period'] = np.mean(holding_periods)
            results['max_holding_period'] = np.max(holding_periods)
            results['min_holding_period'] = np.min(holding_periods)
        else:
            results['win_rate'] = 0
            results['avg_trade_return'] = 0
            results['max_trade_return'] = 0
            results['min_trade_return'] = 0
            results['trade_return_std'] = 0
            results['avg_holding_period'] = 0
            results['max_holding_period'] = 0
            results['min_holding_period'] = 0
            
        return results
    
    def generate_individual_signals_report(self, currency: str, results: dict) -> str:
        """Generate report for individual signal strategies"""
        report = f"""
Individual Signal Strategy Analysis for {currency}
{'='*50}

Buy & Hold Return: {results['EMA_Cross']['buy_and_hold_return']:.2f}%

Performance by Signal Type:
-------------------------"""
        
        for strategy_name, strategy_results in results.items():
            report += f"""

{strategy_name}:
{'-'*len(strategy_name)}
Final Capital: ${strategy_results['final_capital']:,.2f}
Total Return: {strategy_results['strategy_return']:.2f}%
vs Buy & Hold: {strategy_results['strategy_return'] - strategy_results['buy_and_hold_return']:.2f}%
Total Trades: {strategy_results['total_trades']}
Win Rate: {strategy_results['win_rate']:.2f}%
Average Trade Return: {strategy_results['avg_trade_return']:.2f}%
Best Trade: {strategy_results['max_trade_return']:.2f}%
Worst Trade: {strategy_results['min_trade_return']:.2f}%
Average Hold: {strategy_results['avg_holding_period']:.1f} days"""
            
        return report

    def generate_report(self, currency: str, results: dict) -> str:
        """Generate detailed report of backtest results"""
        report = f"""
Sensitive Strategy Analysis Results for {currency}
{'='*50}

Performance Summary:
------------------
Initial Capital: ${results['initial_capital']:,.2f}
Final Capital: ${results['final_capital']:,.2f}
Total Return: {results['strategy_return']:.2f}%
Buy & Hold Return: {results['buy_and_hold_return']:.2f}%
Strategy vs Buy & Hold: {results['strategy_return'] - results['buy_and_hold_return']:.2f}%

Trade Statistics:
---------------
Total Trades: {results['total_trades']}
Profitable Trades: {results['profitable_trades']}
Win Rate: {results['win_rate']:.2f}%
Average Trade Return: {results['avg_trade_return']:.2f}%
Best Trade: {results['max_trade_return']:.2f}%
Worst Trade: {results['min_trade_return']:.2f}%
Trade Return Std Dev: {results['trade_return_std']:.2f}%

Holding Periods:
--------------
Average Hold: {results['avg_holding_period']:.1f} days
Longest Hold: {results['max_holding_period']} days
Shortest Hold: {results['min_holding_period']} days

Detailed Trade History:
--------------------"""

        for i, trade in enumerate(results['trades'], 1):
            report += f"""
Trade {i}:
  Entry: {trade['entry_date'].strftime('%Y-%m-%d')} at ${trade['entry_price']:.2f}
  Exit: {trade['exit_date'].strftime('%Y-%m-%d')} at ${trade['exit_price']:.2f}
  Return: {trade['return_pct']:.2f}%
  Holding Period: {trade['holding_period']} days
  Capital After Trade: ${trade['capital_after_trade']:.2f}"""
        
        return report
    
    def plot_results(self, currency: str, df: pd.DataFrame, results: dict):
        """Create visualization of backtest results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 20))
        gs = plt.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])
        
        # Price and signals plot
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])  # RSI
        ax3 = plt.subplot(gs[2])  # MACD
        ax4 = plt.subplot(gs[3])  # Stochastic
        ax5 = plt.subplot(gs[4])  # Portfolio Value
        
        # Price plot with EMAs
        ax1.plot(df.index, df['price_usd'], label='Price', color='blue', linewidth=1.5)
        ax1.plot(df.index, df['ema_10'], label='10 EMA', color='orange', alpha=0.7)
        ax1.plot(df.index, df['ema_20'], label='20 EMA', color='red', alpha=0.7)
        
        # Plot buy/sell signals
        buy_points = df[df['sensitive_buy']]['price_usd']
        sell_points = df[df['sensitive_sell']]['price_usd']
        ax1.scatter(buy_points.index, buy_points, color='green', marker='^', 
                   s=100, label='Buy Signal')
        ax1.scatter(sell_points.index, sell_points, color='red', marker='v', 
                   s=100, label='Sell Signal')
        
        ax1.set_title(f'{currency} Price and Trading Signals')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # RSI plot
        ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
        ax2.plot(df.index, df['volume_weighted_rsi'], label='Volume-weighted RSI', 
                color='blue', alpha=0.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(df.index, 70, 100, color='r', alpha=0.1)
        ax2.fill_between(df.index, 0, 30, color='g', alpha=0.1)
        ax2.set_title('RSI and Volume-weighted RSI')
        ax2.legend()
        ax2.grid(True)
        
        # MACD plot
        ax3.plot(df.index, df['macd'], label='MACD', color='blue')
        ax3.plot(df.index, df['macd_signal'], label='Signal', color='orange')
        ax3.bar(df.index, df['macd_hist'], label='Histogram', color='gray', alpha=0.3)
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True)
        
        # Stochastic plot
        ax4.plot(df.index, df['k_percent'], label='%K', color='blue')
        ax4.plot(df.index, df['d_percent'], label='%D', color='orange')
        ax4.axhline(y=80, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=20, color='g', linestyle='--', alpha=0.5)
        ax4.fill_between(df.index, 80, 100, color='r', alpha=0.1)
        ax4.fill_between(df.index, 0, 20, color='g', alpha=0.1)
        ax4.set_title('Stochastic Oscillator')
        ax4.legend()
        ax4.grid(True)
        
        # Portfolio value plot
        portfolio_values = [x['capital'] for x in results['price_history']]
        ax5.plot(df.index, portfolio_values, label='Portfolio Value', color='green')
        ax5.set_title('Portfolio Value')
        ax5.legend()
        ax5.grid(True)
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{currency}_analysis_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

def run_sensitive_analysis():
    """Main function to run the sensitive strategy analysis"""
    print("Starting Sensitive Strategy Analysis")
    print("===================================")
    
    backtest = SensitiveStrategyBacktest(DB_CONFIG)
    
    # Test period
    start_date = '2024-01-01'
    end_date = '2025-01-01'
    
    # Create timestamp for files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(backtest.results_dir, f'sensitive_strategy_results_{timestamp}.txt')
    
    # Track overall statistics
    overall_stats = {
        'currencies_tested': 0,
        'total_return': 0,
        'total_trades': 0,
        'profitable_trades': 0,
        'avg_holding_period': 0,
        'best_performer': ('none', -float('inf')),
        'worst_performer': ('none', float('inf'))
    }
    
    with open(results_file, 'w') as f:
        f.write("Sensitive Strategy Analysis Results\n")
        f.write("=================================\n\n")
        f.write(f"Test Period: {start_date} to {end_date}\n\n")
        
        for currency in CURRENCIES:
            print(f"\nProcessing {currency}...")
            try:
                # Load and prepare data
                df = backtest.load_data_from_db(currency, start_date, end_date)
                if df.empty:
                    print(f"No data available for {currency}")
                    continue
                
                # Add indicators and generate signals
                df = backtest.add_technical_indicators(df)
                df = backtest.generate_sensitive_signals(df)
                df = backtest.generate_individual_signals(df)
                
                # Run backtests
                combined_results = backtest.backtest_strategy(df)
                individual_results = backtest.backtest_individual_strategies(df)
                
                # Generate plots
                plot_path = backtest.plot_results(currency, df, combined_results)
                
                # Update overall statistics for combined strategy
                overall_stats['currencies_tested'] += 1
                overall_stats['total_return'] += combined_results['strategy_return']
                overall_stats['total_trades'] += combined_results['total_trades']
                overall_stats['profitable_trades'] += combined_results['profitable_trades']
                overall_stats['avg_holding_period'] += combined_results['avg_holding_period']
                
                if combined_results['strategy_return'] > overall_stats['best_performer'][1]:
                    overall_stats['best_performer'] = (currency, combined_results['strategy_return'])
                if combined_results['strategy_return'] < overall_stats['worst_performer'][1]:
                    overall_stats['worst_performer'] = (currency, combined_results['strategy_return'])

                # Track performance of individual strategies
                for strategy_name, strategy_results in individual_results.items():
                    if strategy_name not in overall_stats:
                        overall_stats[strategy_name] = {
                            'total_return': 0,
                            'total_trades': 0,
                            'profitable_trades': 0,
                            'best_return': (-float('inf'), ''),
                            'worst_return': (float('inf'), '')
                        }
                    
                    overall_stats[strategy_name]['total_return'] += strategy_results['strategy_return']
                    overall_stats[strategy_name]['total_trades'] += strategy_results['total_trades']
                    overall_stats[strategy_name]['profitable_trades'] += strategy_results['profitable_trades']
                    
                    if strategy_results['strategy_return'] > overall_stats[strategy_name]['best_return'][0]:
                        overall_stats[strategy_name]['best_return'] = (strategy_results['strategy_return'], currency)
                    if strategy_results['strategy_return'] < overall_stats[strategy_name]['worst_return'][0]:
                        overall_stats[strategy_name]['worst_return'] = (strategy_results['strategy_return'], currency)
                
                # Generate and write reports
                combined_report = backtest.generate_report(currency, combined_results)
                individual_report = backtest.generate_individual_signals_report(currency, individual_results)
                
                f.write(combined_report)
                f.write("\n\n")
                f.write(individual_report)
                f.write("\n\n" + "="*50 + "\n\n")
                
                print(f"✓ Successfully processed {currency}")
                print(f"  Combined Return: {combined_results['strategy_return']:.2f}%")
                print(f"  Combined Trades: {combined_results['total_trades']}")
                print(f"  Win Rate: {combined_results['win_rate']:.2f}%")
                print(f"  Plot saved to: {plot_path}")
                
            except Exception as e:
                print(f"Error processing {currency}: {str(e)}")
                continue
        
        # Write overall summary
        if overall_stats['currencies_tested'] > 0:
            summary = f"""
Overall Strategy Performance
==========================
Currencies Tested: {overall_stats['currencies_tested']}

Combined Strategy:
---------------
Average Return: {overall_stats['total_return'] / overall_stats['currencies_tested']:.2f}%
Total Trades: {overall_stats['total_trades']}
Overall Win Rate: {(overall_stats['profitable_trades'] / overall_stats['total_trades'] * 100) if overall_stats['total_trades'] > 0 else 0:.2f}%
Average Holding Period: {overall_stats['avg_holding_period'] / overall_stats['currencies_tested']:.1f} days
Best Performer: {overall_stats['best_performer'][0]} ({overall_stats['best_performer'][1]:.2f}%)
Worst Performer: {overall_stats['worst_performer'][0]} ({overall_stats['worst_performer'][1]:.2f}%)

Individual Strategies Performance:
------------------------------"""
            
            for strategy_name in ['EMA_Cross', 'RSI', 'MACD', 'Stochastic', 'Volume_RSI']:
                if strategy_name in overall_stats:
                    stats = overall_stats[strategy_name]
                    trades = stats['total_trades']
                    win_rate = (stats['profitable_trades'] / trades * 100) if trades > 0 else 0
                    avg_return = stats['total_return'] / overall_stats['currencies_tested']
                    
                    summary += f"""
{strategy_name}:
  Average Return: {avg_return:.2f}%
  Total Trades: {trades}
  Win Rate: {win_rate:.2f}%
  Best: {stats['best_return'][1]} ({stats['best_return'][0]:.2f}%)
  Worst: {stats['worst_return'][1]} ({stats['worst_return'][0]:.2f}%)"""

            f.write(summary)
            print("\n" + summary)
        else:
            summary = "\nNo successful tests completed.\n"
            f.write(summary)
            print(summary)
    
    print(f"\nAnalysis completed!")
    print(f"Results saved to: {results_file}")
    print(f"Plots saved in: {backtest.plots_dir}")

if __name__ == "__main__":
    run_sensitive_analysis()
