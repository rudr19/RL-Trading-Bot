import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
import torch

# Set page config
st.set_page_config(
    page_title="🤖 RL Stock Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .error-text {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-text {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Import the trading environment class
class StockTradingEnvironment(gym.Env):
    """
    Custom Stock Trading Environment for Reinforcement Learning
    (Same as in the Colab notebook)
    """
    
    def __init__(self, df, initial_balance=10000, max_position_size=0.3, 
                 stop_loss=0.05, take_profit=0.1, transaction_cost=0.001):
        super(StockTradingEnvironment, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.transaction_cost = transaction_cost
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: technical indicators + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        self.reset()
    
    def _calculate_technical_indicators(self, data):
        """Calculate technical indicators for the current state"""
        if len(data) < 20:
            return np.zeros(8)
        
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        # Simple Moving Averages
        sma_5 = np.mean(close[-5:])
        sma_20 = np.mean(close[-20:])
        
        # RSI (Relative Strength Index)
        delta = np.diff(close[-15:])
        gain = np.mean(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
        loss = np.mean(-delta[delta < 0]) if len(delta[delta < 0]) > 0 else 0
        rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # MACD
        ema_12 = close[-1] * (2/13) + close[-2] * (1 - 2/13) if len(close) > 1 else close[-1]
        ema_26 = close[-1] * (2/27) + close[-2] * (1 - 2/27) if len(close) > 1 else close[-1]
        macd = ema_12 - ema_26
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = np.std(close[-20:])
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Volume indicator
        volume_sma = np.mean(volume[-5:])
        volume_ratio = volume[-1] / (volume_sma + 1e-8)
        
        # Price momentum
        momentum = (close[-1] - close[-5]) / (close[-5] + 1e-8) if len(close) >= 5 else 0
        
        # Volatility
        volatility = np.std(close[-10:]) / (np.mean(close[-10:]) + 1e-8) if len(close) >= 10 else 0
        
        return np.array([
            (close[-1] - sma_5) / (sma_5 + 1e-8),
            (close[-1] - sma_20) / (sma_20 + 1e-8),
            rsi / 100.0,
            macd / (close[-1] + 1e-8),
            bb_position,
            volume_ratio,
            momentum,
            volatility
        ])
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step < 20:
            technical_indicators = np.zeros(8)
        else:
            data_slice = self.df.iloc[max(0, self.current_step-20):self.current_step+1]
            technical_indicators = self._calculate_technical_indicators(data_slice)
        
        # Portfolio state
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.shares_held / (self.initial_balance / current_price),
            (current_price - self.avg_buy_price) / (self.avg_buy_price + 1e-8) if self.avg_buy_price > 0 else 0,
            (self.net_worth - self.initial_balance) / self.initial_balance
        ])
        
        return np.concatenate([technical_indicators, portfolio_state]).astype(np.float32)
    
    def reset(self, seed=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 20
        self.balance = self.initial_balance
        self.shares_held = 0
        self.avg_buy_price = 0
        self.net_worth = self.initial_balance
        self.trades = []
        self.portfolio_values = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one time step within the environment"""
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Calculate current net worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.portfolio_values.append(self.net_worth)
        
        # Risk management checks
        if self.shares_held > 0:
            unrealized_return = (current_price - self.avg_buy_price) / self.avg_buy_price
            
            # Stop loss
            if unrealized_return <= -self.stop_loss:
                action = 2  # Force sell
            
            # Take profit
            elif unrealized_return >= self.take_profit:
                action = 2  # Force sell
        
        # Execute action
        reward = 0
        
        if action == 1:  # Buy
            max_shares = int((self.balance * self.max_position_size) / current_price)
            if max_shares > 0 and self.balance > current_price:
                shares_to_buy = min(max_shares, int(self.balance / current_price))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= self.balance:
                    self.avg_buy_price = ((self.avg_buy_price * self.shares_held) + 
                                        (current_price * shares_to_buy)) / (self.shares_held + shares_to_buy)
                    self.shares_held += shares_to_buy
                    self.balance -= cost
                    
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'total_cost': cost
                    })
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                sell_value = self.shares_held * current_price * (1 - self.transaction_cost)
                
                if self.avg_buy_price > 0:
                    profit_pct = (current_price - self.avg_buy_price) / self.avg_buy_price
                    reward = profit_pct * 10
                
                self.balance += sell_value
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': self.shares_held,
                    'total_value': sell_value
                })
                
                self.shares_held = 0
                self.avg_buy_price = 0
        
        # Calculate step reward
        if len(self.portfolio_values) > 1:
            portfolio_return = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            reward += portfolio_return * 100
        
        if action == 0:  # Hold penalty
            reward -= 0.01
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        final_net_worth = self.balance + (self.shares_held * current_price)
        
        return self._get_observation(), reward, done, False, {
            'net_worth': final_net_worth,
            'trades': len(self.trades),
            'balance': self.balance,
            'shares_held': self.shares_held
        }

# Utility functions
@st.cache_data
def download_stock_data(symbol, period='1y'):
    """Download stock data with caching"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error downloading data for {symbol}: {str(e)}")
        return None

def calculate_performance_metrics(portfolio_values, initial_value=10000):
    """Calculate trading performance metrics"""
    if len(portfolio_values) < 2:
        return {"error": "Insufficient data"}
    
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    cumulative_return = (portfolio_values[-1] - initial_value) / initial_value
    
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
    else:
        sharpe_ratio = 0
    
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    
    return {
        'total_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_value': portfolio_values[-1],
        'total_trades': len(returns)
    }

def plot_portfolio_performance(portfolio_values, trades, stock_data):
    """Create portfolio performance plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Value Over Time', 'Stock Price with Trades', 
                       'Returns Distribution', 'Drawdown'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(y=portfolio_values, name='Portfolio Value', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Stock price with trades
    fig.add_trace(
        go.Scatter(y=stock_data['Close'], name='Stock Price', line=dict(color='black')),
        row=1, col=2
    )
    
    # Add trade markers
    buy_points = [t for t in trades if t['action'] == 'BUY']
    sell_points = [t for t in trades if t['action'] == 'SELL']
    
    if buy_points:
        buy_steps = [t['step'] for t in buy_points]
        buy_prices = [t['price'] for t in buy_points]
        fig.add_trace(
            go.Scatter(x=buy_steps, y=buy_prices, mode='markers', 
                      name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)),
            row=1, col=2
        )
    
    if sell_points:
        sell_steps = [t['step'] for t in sell_points]
        sell_prices = [t['price'] for t in sell_points]
        fig.add_trace(
            go.Scatter(x=sell_steps, y=sell_prices, mode='markers', 
                      name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)),
            row=1, col=2
        )
    
    # Returns distribution
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        fig.add_trace(
            go.Histogram(x=returns, name='Returns', nbinsx=30, marker=dict(color='purple')),
            row=2, col=1
        )
    
    # Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / peak
    fig.add_trace(
        go.Scatter(y=drawdown, fill='tonegative', name='Drawdown', 
                  fillcolor='rgba(255,0,0,0.3)', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Trading Performance Analysis")
    return fig

def run_backtest(model, stock_data, algorithm_name):
    """Run backtest with the loaded model"""
    env = StockTradingEnvironment(stock_data)
    
    obs, _ = env.reset()
    done = False
    actions_taken = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    step_count = 0
    total_steps = len(stock_data) - 21
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        actions_taken.append(action)
        
        step_count += 1
        progress = min(step_count / total_steps, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Backtesting... Step {step_count}/{total_steps}")
    
    progress_bar.empty()
    status_text.empty()
    
    return env.portfolio_values, env.trades, actions_taken, info

def get_current_prediction(model, stock_data):
    """Get current trading decision from the model"""
    if len(stock_data) < 21:
        return "HOLD", 0.33
    
    env = StockTradingEnvironment(stock_data.tail(50))  # Use recent data
    obs, _ = env.reset()
    
    # Get the latest observation
    env.current_step = len(env.df) - 1
    obs = env._get_observation()
    
    action, _ = model.predict(obs, deterministic=True)
    
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action_colors = {0: 0.33, 1: 0.0, 2: 1.0}  # For color coding
    
    return action_names[action], action_colors[action]

# Main Streamlit App
def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RL Stock Trading Bot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model upload or selection
    st.sidebar.subheader("📁 Model Selection")
    
    model_source = st.sidebar.radio(
        "Choose model source:",
        ["Upload Model File", "Use Sample Models"]
    )
    
    uploaded_model = None
    selected_algorithm = None
    
    if model_source == "Upload Model File":
        uploaded_model = st.sidebar.file_uploader(
            "Upload trained model (.zip file)", 
            type=['zip'],
            help="Upload a model trained using the Google Colab notebook"
        )
        
        if uploaded_model:
            # Save uploaded file temporarily
            with open("temp_model.zip", "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            # Detect algorithm from filename
            if "DQN" in uploaded_model.name.upper():
                selected_algorithm = "DQN"
            elif "PPO" in uploaded_model.name.upper():
                selected_algorithm = "PPO"
            else:
                selected_algorithm = st.sidebar.selectbox("Select Algorithm:", ["DQN", "PPO"])
    
    else:
        st.sidebar.info("📝 Note: Sample models are for demonstration. Upload your own trained models for better performance.")
        selected_algorithm = st.sidebar.selectbox("Select Algorithm:", ["DQN", "PPO"])
    
    # Stock selection
    st.sidebar.subheader("📈 Stock Selection")
    
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Enter Ticker Symbol", "Upload CSV Data"]
    )
    
    stock_data = None
    symbol = None
    
    if input_method == "Enter Ticker Symbol":
        symbol = st.sidebar.text_input(
            "Stock Ticker Symbol", 
            value="AAPL",
            help="Enter a valid stock ticker (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        period = st.sidebar.selectbox(
            "Data Period:",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=3
        )
        
        if symbol:
            with st.spinner(f"Downloading data for {symbol}..."):
                stock_data = download_stock_data(symbol, period)
    
    else:
        uploaded_csv = st.sidebar.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="CSV should have columns: Date, Open, High, Low, Close, Volume"
        )
        
        if uploaded_csv:
            try:
                stock_data = pd.read_csv(uploaded_csv)
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                symbol = "CUSTOM"
                st.sidebar.success("✅ CSV data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading CSV: {str(e)}")
    
    # Trading parameters
    st.sidebar.subheader("⚙️ Trading Parameters")
    initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000, min_value=1000)
    max_position_size = st.sidebar.slider("Max Position Size (%)", min_value=10, max_value=100, value=30) / 100
    stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=1, max_value=20, value=5) / 100
    take_profit = st.sidebar.slider("Take Profit (%)", min_value=5, max_value=50, value=10) / 100
    
    # Main content area
    if stock_data is not None and len(stock_data) > 21:
        
        # Display stock info
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
        price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}")
        
        with col2:
            st.metric("Price Change %", f"{price_change_pct:+.2f}%")
        
        with col3:
            st.metric("Data Points", len(stock_data))
        
        with col4:
            st.metric("Date Range", f"{stock_data['Date'].min().strftime('%Y-%m-%d')} to {stock_data['Date'].max().strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        
        # Model loading and prediction section
        st.subheader("🤖 AI Trading Decisions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Run Backtest", type="primary"):
                if uploaded_model or model_source == "Use Sample Models":
                    
                    try:
                        # Load model
                        with st.spinner("Loading model..."):
                            if uploaded_model:
                                if selected_algorithm == "DQN":
                                    model = DQN.load("temp_model.zip")
                                else:
                                    model = PPO.load("temp_model.zip")
                            else:
                                # Create a dummy model for demonstration
                                env = StockTradingEnvironment(stock_data.head(100))
                                if selected_algorithm == "DQN":
                                    model = DQN('MlpPolicy', env)
                                else:
                                    model = PPO('MlpPolicy', env)
                                st.warning("⚠️ Using untrained sample model. Results may not be meaningful.")
                        
                        # Run backtest
                        st.success("✅ Model loaded successfully!")
                        
                        with st.spinner("Running backtest..."):
                            portfolio_values, trades, actions, final_info = run_backtest(model, stock_data, selected_algorithm)
                        
                        # Store results in session state
                        st.session_state['backtest_results'] = {
                            'portfolio_values': portfolio_values,
                            'trades': trades,
                            'actions': actions,
                            'final_info': final_info,
                            'stock_data': stock_data,
                            'symbol': symbol,
                            'algorithm': selected_algorithm
                        }
                        
                        st.success("✅ Backtest completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Error running backtest: {str(e)}")
                
                else:
                    st.warning("⚠️ Please upload a model file first.")
        
        with col2:
            if st.button("🎯 Get Current Signal"):
                if uploaded_model or model_source == "Use Sample Models":
                    try:
                        # Load model
                        if uploaded_model:
                            if selected_algorithm == "DQN":
                                model = DQN.load("temp_model.zip")
                            else:
                                model = PPO.load("temp_model.zip")
                        else:
                            # Create dummy model
                            env = StockTradingEnvironment(stock_data.head(100))
                            if selected_algorithm == "DQN":
                                model = DQN('MlpPolicy', env)
                            else:
                                model = PPO('MlpPolicy', env)
                        
                        # Get current prediction
                        action, color_val = get_current_prediction(model, stock_data)
                        
                        # Display prediction
                        if action == "BUY":
                            st.success(f"🟢 **{action}** - Model suggests buying")
                        elif action == "SELL":
                            st.error(f"🔴 **{action}** - Model suggests selling")
                        else:
                            st.info(f"🟡 **{action}** - Model suggests holding")
                    
                    except Exception as e:
                        st.error(f"❌ Error getting prediction: {str(e)}")
                else:
                    st.warning("⚠️ Please upload a model file first.")
        
        # Display backtest results if available
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            
            st.markdown("---")
            st.subheader("📊 Backtest Results")
            
            # Calculate metrics
            metrics = calculate_performance_metrics(results['portfolio_values'], initial_balance)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                color = "success" if metrics['total_return'] > 0 else "error"
                st.metric("Total Return", f"{metrics['total_return']:.2%}", 
                         delta=f"${metrics['final_value'] - initial_balance:.2f}")
            
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            
            with col4:
                st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
            
            # Performance plots
            st.subheader("📈 Performance Analysis")
            
            fig = plot_portfolio_performance(
                results['portfolio_values'], 
                results['trades'], 
                results['stock_data']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading summary
            st.subheader("📋 Trading Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Trade Statistics:**")
                buy_trades = [t for t in results['trades'] if t['action'] == 'BUY']
                sell_trades = [t for t in results['trades'] if t['action'] == 'SELL']
                
                st.write(f"- Total Trades: {len(results['trades'])}")
                st.write(f"- Buy Orders: {len(buy_trades)}")
                st.write(f"- Sell Orders: {len(sell_trades)}")
                st.write(f"- Final Balance: ${results['final_info']['balance']:.2f}")
                st.write(f"- Shares Held: {results['final_info']['shares_held']}")
            
            with col2:
                st.write("**Algorithm Performance:**")
                st.write(f"- Algorithm Used: {results['algorithm']}")
                st.write(f"- Initial Investment: ${initial_balance:,.2f}")
                st.write(f"- Final Portfolio Value: ${metrics['final_value']:,.2f}")
                st.write(f"- Risk Management: Stop Loss {stop_loss:.1%}, Take Profit {take_profit:.1%}")
            
            # Recent trades table
            if results['trades']:
                st.subheader("🔄 Recent Trades")
                trades_df = pd.DataFrame(results['trades'])
                trades_df['date'] = stock_data.iloc[trades_df['step']]['Date'].values
                trades_df = trades_df[['date', 'action', 'price', 'shares', 'total_cost', 'total_value']].fillna(0)
                st.dataframe(trades_df.tail(10), use_container_width=True)
    
    else:
        # Instructions when no data is loaded
        st.info("👋 Welcome to the RL Stock Trading Bot!")
        
        st.markdown("""
        ### 🚀 Getting Started:
        
        1. **Train a model** using the Google Colab notebook
        2. **Upload your trained model** (.zip file) in the sidebar
        3. **Select a stock** ticker or upload CSV data
        4. **Configure trading parameters** (stop-loss, take-profit, etc.)
        5. **Run backtest** to see how your model performs
        6. **Get real-time signals** for current market conditions
        
        ### 📚 Features:
        - **Multiple RL Algorithms**: DQN and PPO support
        - **Risk Management**: Built-in stop-loss and take-profit
        - **Performance Metrics**: Sharpe ratio, drawdown, win rate
        - **Interactive Charts**: Portfolio performance and trade visualization
        - **Real-time Predictions**: Get current buy/sell/hold signals
        
        ### 🔧 Requirements:
        - Trained model file (.zip format)
        - Valid stock ticker or CSV data with OHLCV columns
        
        Start by uploading a model or entering a stock symbol in the sidebar! 👈
        """)
        
        # Sample data preview
        st.subheader("📊 Sample Data Format")
        sample_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [150.00, 151.00, 149.00],
            'High': [152.00, 153.00, 150.00],
            'Low': [149.50, 150.00, 148.00],
            'Close': [151.00, 149.50, 149.75],
            'Volume': [1000000, 1200000, 950000]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

if __name__ == "__main__":
    main()