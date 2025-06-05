# 🤖 RL Stock Trading Bot – Professional Deployment Guide

A **state-of-the-art Reinforcement Learning-based stock trading bot** built with **DQN** and **PPO** algorithms. Comes with a modern **Streamlit-based interface** for effortless backtesting, visualization, and real-time trading signal generation.

---

## 🚀 Quick Start

### 🔬 Step 1: Train Model in Google Colab
1. Open `colab_training.ipynb` in [Google Colab](https://colab.research.google.com)
2. Train the model with your chosen stock symbol (e.g., `AAPL`)
3. Download the generated `.zip` model and `.pkl` result files

---

### 💻 Step 2: Deploy Locally with Streamlit
```bash
# Clone the repo
git clone <your-repo-url>
cd rl-stock-trading-bot

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app.py
```

---

### ☁️ Step 3: Deploy on Cloud

#### ✅ Option A: [Streamlit Cloud](https://share.streamlit.io)
1. Push your code to GitHub  
2. Visit [Streamlit Cloud](https://share.streamlit.io)  
3. Connect your GitHub repo  
4. Deploy instantly ⚡

#### ♻️ Option B: Heroku
```bash
# Prepare Heroku deployment
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

heroku create your-app-name
git add .
git commit -m "Deploy RL trading bot"
git push heroku main
```

#### 🚀 Option C: Render
1. Connect your GitHub repo to [Render](https://render.com)
2. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

---

## 📋 Project Structure
```
rl-stock-trading-bot/
├── colab_training.ipynb          # Colab notebook for training
├── streamlit_app.py              # Web interface using Streamlit
├── requirements.txt              # Required Python packages
├── models/                       # Trained model files
│   ├── AAPL_DQN_model.zip
│   └── AAPL_PPO_model.zip
├── data/
│   └── sample_data.csv           # Example stock data
└── README.md
```

---

## 🔧 Features

### 🧰 Core Capabilities
- **Dual Algorithms**: DQN and PPO using Stable-Baselines3
- **Custom OpenAI Gym Environment**: Tailored to trading logic
- **Risk Control**: Stop-loss, take-profit, position sizing
- **Technical Indicators**: RSI, MACD, Bollinger Bands, MAs
- **Performance Metrics**: Sharpe Ratio, Drawdown, Win Rate

### 🌐 Streamlit Interface
- Upload trained models
- Select stocks via ticker or CSV upload
- View live Buy/Sell/Hold signals
- Visualize performance with dynamic charts
- Run historical backtests

### ⚡ Risk Management Tools
- Adjustable stop-loss/take-profit levels
- Max position size control
- Simulated transaction cost support

---

## 📊 How to Use

### 📈 Train in Colab
```python
STOCK_SYMBOL = 'AAPL'
ALGORITHM = 'DQN'       # Options: 'DQN', 'PPO'
TRAINING_TIMESTEPS = 50000
```
Run all cells and download the model files.

### 📹 Streamlit Web App
1. Upload a `.zip` trained model
2. Choose a stock symbol or upload CSV
3. Adjust risk settings
4. Run backtest and analyze results
5. Generate current signals

### 📆 CSV Format Example
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.00,152.00,149.50,151.00,1000000
2024-01-02,151.00,153.00,150.00,149.50,1200000
```

---

## 🛠️ Customization

### ⚙️ Trading Environment
Update `StockTradingEnvironment` class to:
- Add indicators or signals
- Change reward structure
- Modify action space
- Alter trading rules

### 🔄 Training Parameters
```python
# DQN
learning_rate=0.0003
buffer_size=10000
learning_starts=1000
target_update_interval=1000

# PPO
learning_rate=0.0003
n_steps=2048
batch_size=64
n_epochs=10
```

### ⛓ Risk Config
```python
initial_balance=10000
max_position_size=0.3
stop_loss=0.05
take_profit=0.1
transaction_cost=0.001
```

---

## 📊 Performance Metrics
- **Total Return**
- **Sharpe Ratio**
- **Max Drawdown**
- **Win Rate**
- **Volatility**

---

## ⚠️ Important Notes
- This tool is for **educational purposes** only
- Historical data doesn't guarantee future results
- Always validate on **out-of-sample data**
- Enable **paper trading** before real trading

---

## 🛥️ Troubleshooting

#### 🚫 Model Load Error
```python
if uploaded_model.name.endswith('.zip'):
    model = DQN.load(uploaded_model)
```

#### 🚫 Invalid Ticker
```python
stock = yf.Ticker(symbol)
if 'regularMarketPrice' not in stock.info:
    st.error("Invalid ticker symbol")
```

#### 🧠 Memory Issues
- Use fewer training steps for testing
- Minimize dataset size
- Restart Streamlit session

---

## 📞 Support
1. Review this README and notebook comments
2. Try sample data and verify setup
3. Double-check Python environment and package versions

---

## 📄 License
This project is open-source and for learning purposes. Use at your own discretion.

---

## ♻️ Changelog

### Version 1.0
- Base support for PPO & DQN
- Risk control system
- Web UI with model upload and testing

### Planned Upgrades
- [ ] Multi-asset portfolio support
- [ ] Real-time feeds & notifications
- [ ] Mobile-first UI
- [ ] Enhanced indicator library
- [ ] Model ensembles for robustness

---

**Happy Trading! 📈🚀**

*Disclaimer: Always conduct your own research. Invest responsibly.*

