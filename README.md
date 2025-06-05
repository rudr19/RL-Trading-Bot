# 🤖 RL Stock Trading Bot - Complete Deployment Guide

A sophisticated Reinforcement Learning-based stock trading bot using DQN and PPO algorithms with a user-friendly Streamlit interface.

## 🚀 Quick Start

### 1. Google Colab Training
1. Open the `colab_training.ipynb` notebook in Google Colab
2. Run all cells to train your model on AAPL (or modify the `STOCK_SYMBOL` variable)
3. Download the generated `.zip` model file and `.pkl` results file

### 2. Local Streamlit Deployment
```bash
# Clone or download the project files
git clone <your-repo-url>
cd rl-stock-trading-bot

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### 3. Cloud Deployment Options

#### Option A: Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

#### Option B: Heroku Deployment
```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
git add .
git commit -m "Deploy RL trading bot"
git push heroku main
```

#### Option C: Render Deployment
1. Connect your GitHub repository to Render
2. Use these build settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

## 📋 Project Structure

```
rl-stock-trading-bot/
├── colab_training.ipynb          # Google Colab training notebook
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── models/                       # Directory for trained models
│   ├── AAPL_DQN_model.zip       # Example trained DQN model
│   └── AAPL_PPO_model.zip       # Example trained PPO model
└── data/                         # Directory for data files
    └── sample_data.csv           # Sample stock data format
```

## 🔧 Features

### Core Functionality
- **Multiple RL Algorithms**: DQN and PPO implementations
- **Custom Trading Environment**: OpenAI Gym compatible
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Risk Management**: Stop-loss, take-profit, position sizing
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate

### Web Interface
- **Model Upload**: Support for trained .zip model files
- **Stock Selection**: Ticker input or CSV upload
- **Real-time Signals**: Current buy/sell/hold recommendations
- **Interactive Charts**: Portfolio performance visualization
- **Backtesting**: Historical performance analysis

### Risk Management Features
- **Stop-Loss**: Configurable percentage-based stop-loss
- **Take-Profit**: Automatic profit-taking at target levels
- **Position Sizing**: Maximum position size limits
- **Transaction Costs**: Realistic trading cost simulation

## 📊 Usage Instructions

### Training Models (Google Colab)
1. Open the Colab notebook
2. Modify parameters as needed:
   ```python
   STOCK_SYMBOL = 'AAPL'  # Change to any stock
   ALGORITHM = 'DQN'      # 'DQN' or 'PPO'
   TRAINING_TIMESTEPS = 50000
   ```
3. Run all cells
4. Download the generated model files

### Using the Streamlit App
1. **Upload Model**: Use the sidebar to upload your trained .zip model
2. **Select Stock**: Enter a ticker symbol or upload CSV data
3. **Configure Parameters**: Set risk management settings
4. **Run Backtest**: Analyze historical performance
5. **Get Signals**: Get current trading recommendations

### CSV Data Format
Your CSV file should have these columns:
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.00,152.00,149.50,151.00,1000000
2024-01-02,151.00,153.00,150.00,149.50,1200000
```

## 🛠️ Customization

### Modifying the Trading Environment
Edit the `StockTradingEnvironment` class to:
- Add new technical indicators
- Modify reward function
- Change action space (e.g., add position sizing)
- Adjust risk management rules

### Training Parameters
Key parameters to tune:
```python
# DQN Parameters
learning_rate=0.0003
buffer_size=10000
learning_starts=1000
target_update_interval=1000

# PPO Parameters
learning_rate=0.0003
n_steps=2048
batch_size=64
n_epochs=10
```

### Risk Management
Adjust risk parameters:
```python
initial_balance=10000      # Starting capital
max_position_size=0.3      # Max 30% position size
stop_loss=0.05            # 5% stop loss
take_profit=0.1           # 10% take profit
transaction_cost=0.001    # 0.1% transaction cost
```

## 📈 Performance Metrics

The bot calculates several key metrics:

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Volatility**: Portfolio volatility measurement

## ⚠️ Important Notes

### Limitations
- **Not Financial Advice**: This is for educational purposes only
- **Past Performance**: Historical results don't guarantee future performance
- **Market Conditions**: Model performance varies with market conditions
- **Transaction Costs**: Real trading involves additional costs and slippage

### Recommendations
- **Paper Trading**: Test thoroughly before using real money
- **Risk Management**: Always use appropriate position sizing
- **Model Validation**: Validate on out-of-sample data
- **Regular Retraining**: Update models with new data periodically

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```python
# Ensure model file is correct format
if uploaded_model.name.endswith('.zip'):
    model = DQN.load(uploaded_model)
```

#### Data Download Issues
```python
# Check ticker symbol validity
stock = yf.Ticker(symbol)
info = stock.info
if 'regularMarketPrice' not in info:
    st.error("Invalid ticker symbol")
```

#### Memory Issues
- Reduce `TRAINING_TIMESTEPS` for faster training
- Use smaller datasets for testing
- Clear browser cache if Streamlit becomes slow

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with sample data first
4. Ensure all dependencies are installed correctly

## 📜 License

This project is for educational purposes. Use at your own risk.

## 🔄 Updates

### Version 1.0
- Initial release with DQN and PPO support
- Basic risk management features
- Streamlit web interface

### Future Enhancements
- [ ] Multi-asset portfolio support
- [ ] Advanced technical indicators
- [ ] Real-time data feeds
- [ ] Model ensemble methods
- [ ] Advanced risk metrics
- [ ] Mobile-responsive design

---

**Happy Trading! 📈🚀**

*Remember: This is for educational purposes only. Always do your own research and never invest more than you can afford to lose.*
