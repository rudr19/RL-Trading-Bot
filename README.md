<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RL Trading Bot</title>
  <style>
    body {
      font-family: 'Arial', sans-serif;
      margin: 0;
      padding: 0;
      background-color: #f5f7fa;
      color: #333;
      text-align: center;
      transition: background-color 0.3s, color 0.3s;
    }
    body.dark {
      background-color: #1a1a1a;
      color: #e0e0e0;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
    }
    h1 {
      font-size: 3em;
      margin: 20px 0;
      animation: fadeIn 1s ease-in;
    }
    .tagline {
      font-style: italic;
      font-size: 1.2em;
      margin-bottom: 20px;
      color: #666;
    }
    .dark .tagline {
      color: #aaa;
    }
    .badges, .tools {
      margin: 20px 0;
    }
    .badges img, .tools img {
      margin: 5px;
      border-radius: 5px;
      transition: transform 0.3s;
      animation: fadeIn 1.5s ease-in;
    }
    .badges img:hover, .tools img:hover {
      transform: scale(1.1);
    }
    .animation {
      margin: 20px 0;
    }
    .section {
      margin: 40px 0;
      text-align: left;
    }
    h2 {
      color: #2c3e50;
      border-bottom: 2px solid #2c3e50;
      padding-bottom: 5px;
    }
    .dark h2 {
      color: #3498db;
      border-bottom-color: #3498db;
    }
    pre, code {
      background-color: #ecf0f1;
      padding: 10px;
      border-radius: 5px;
      overflow-x: auto;
    }
    .dark pre, .dark code {
      background-color: #2c2c2c;
      color: #e0e0e0;
    }
    ul {
      list-style-type: none;
      padding: 0;
    }
    li {
      margin: 10px 0;
    }
    li::before {
      content: "✔️";
      margin-right: 10px;
    }
    a {
      color: #3498db;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .dark a {
      color: #66b0ff;
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 10px 20px;
      background-color: #3498db;
      color: white;
      border: none;
      border-radius: 5px;
      cursor: pointer;
    }
    .dark .theme-toggle {
      background-color: #66b0ff;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  </style>
</head>
<body>
  <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
  <div class="container">
    <h1>RL-TRADING-BOT 🤖💹</h1>
    <p class="tagline">Master the market with intelligent trading strategies.</p>

    <div class="badges">
      <img src="https://img.shields.io/badge/Last%20Commit-today-4A4A4A?style=flat-square&labelColor=000000" alt="Last Commit"/>
      <img src="https://img.shields.io/badge/Jupyter%20Notebook-85.5%25-F37626?style=flat-square&labelColor=000000" alt="Jupyter Notebook"/>
      <img src="https://img.shields.io/badge/Languages-2-1F77B4?style=flat-square&labelColor=000000" alt="Languages"/>
    </div>

    <div class="animation">
      <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOG1ycmQ3N3M4c2o0Nzl0c3V4Y2k0b2RtbjJ0ZGg1NzB0bG9wM3B2bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKz0fYvC8i3lS8Q/giphy.gif" width="300" alt="Stock Market Animation"/>
    </div>

    <p><strong>Built with the tools and technologies:</strong></p>
    <div class="tools">
      <img src="https://img.shields.io/badge/Markdown-000000?style=flat-square&logo=markdown&logoColor=white" alt="Markdown"/>
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
      <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
      <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
      <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
      <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"/>
      <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
    </div>

    <div class="section">
      <h2>🚀 Quick Start</h2>
      <h3>1. Train in Google Colab 🧠</h3>
      <ol>
        <li>Open <a href="./colab_training.ipynb">colab_training.ipynb</a> in <a href="https://colab.research.google.com">Google Colab</a>.</li>
        <li>Set <code>STOCK_SYMBOL = 'AAPL'</code> (or your preferred stock).</li>
        <li>Run all cells to train.</li>
        <li>Download the <code>.zip</code> model and <code>.pkl</code> results.</li>
      </ol>

      <h3>2. Run Locally with Streamlit 💻</h3>
      <pre><code>git clone &lt;your-repo-url&gt;
cd rl-stock-trading-bot
pip install -r requirements.txt
streamlit run streamlit_app.py</code></pre>

      <h3>3. Cloud Deployment Options ☁️</h3>
      <h4>Streamlit Cloud (Recommended)</h4>
      <ol>
        <li>Push your code to GitHub.</li>
        <li>Visit <a href="https://share.streamlit.io">share.streamlit.io</a>.</li>
        <li>Connect your repository and deploy with one click.</li>
      </ol>

      <h4>Heroku</h4>
      <pre><code>echo "web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
heroku create your-app-name
git add .
git commit -m "Deploy RL trading bot"
git push heroku main</code></pre>

      <h4>Render</h4>
      <ol>
        <li>Connect your GitHub repository to <a href="https://render.com">Render</a>.</li>
        <li>Configure:</li>
        <ul>
          <li><strong>Build Command</strong>: <code>pip install -r requirements.txt</code></li>
          <li><strong>Start Command</strong>: <code>streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0</code></li>
        </ul>
      </ol>
    </div>

    <div class="section">
      <h2>📂 Project Structure</h2>
      <pre><code>rl-stock-trading-bot/
├── colab_training.ipynb     # Training notebook for Google Colab
├── streamlit_app.py         # Streamlit application
├── requirements.txt         # Python dependencies
├── models/                  # Trained model files
│   ├── AAPL_DQN_model.zip  # Example DQN model
│   └── AAPL_PPO_model.zip  # Example PPO model
├── data/                    # Stock data files
│   └── sample_data.csv      # Sample stock data
└── README.md                # Project documentation</code></pre>
    </div>

    <div class="section">
      <h2>🔧 Features</h2>
      <h3>Core Functionality</h3>
      <ul>
        <li>Multiple RL Algorithms: DQN and PPO implementations</li>
        <li>Custom Trading Environment: OpenAI Gym compatible</li>
        <li>Technical Indicators: RSI, MACD, Bollinger Bands, Moving Averages</li>
        <li>Risk Management: Stop-loss, take-profit, position sizing</li>
        <li>Performance Metrics: Sharpe ratio, maximum drawdown, win rate</li>
      </ul>

      <h3>Web Interface</h3>
      <ul>
        <li>Model Upload: Support for trained .zip model files</li>
        <li>Stock Selection: Ticker input or CSV upload</li>
        <li>Real-time Signals: Current buy/sell/hold recommendations</li>
        <li>Interactive Charts: Portfolio performance visualization</li>
        <li>Backtesting: Historical performance analysis</li>
      </ul>

      <h3>Risk Management Features</h3>
      <ul>
        <li>Stop-Loss: Configurable percentage-based stop-loss</li>
        <li>Take-Profit: Automatic profit-taking at target levels</li>
        <li>Position Sizing: Maximum position size limits</li>
        <li>Transaction Costs: Realistic trading cost simulation</li>
      </ul>
    </div>

    <div class="section">
      <h2>📊 Usage Instructions</h2>
      <h3>Training Models (Google Colab)</h3>
      <ol>
        <li>Open the Colab notebook.</li>
        <li>Modify parameters as needed:</li>
        <pre><code>STOCK_SYMBOL = 'AAPL'  # Change to any stock
ALGORITHM = 'DQN'      # 'DQN' or 'PPO'
TRAINING_TIMESTEPS = 50000</code></pre>
        <li>Run all cells.</li>
        <li>Download the generated model files.</li>
      </ol>

      <h3>Using the Streamlit App</h3>
      <ol>
        <li><strong>Upload Model</strong>: Use the sidebar to upload your trained .zip model.</li>
        <li><strong>Select Stock</strong>: Enter a ticker symbol or upload CSV data.</li>
        <li><strong>Configure Parameters</strong>: Set risk management settings.</li>
        <li><strong>Run Backtest</strong>: Analyze historical performance.</li>
        <li><strong>Get Signals</strong>: Get current trading recommendations.</li>
      </ol>

      <h3>CSV Data Format</h3>
      <pre><code>Date,Open,High,Low,Close,Volume
2024-01-01,150.00,152.00,149.50,151.00,1000000
2024-01-02,151.00,153.00,150.00,149.50,1200000</code></pre>
    </div>

    <div class="section">
      <h2>🛠️ Customization</h2>
      <h3>Modifying the Trading Environment</h3>
      <p>Edit the <code>StockTradingEnvironment</code> class to:</p>
      <ul>
        <li>Add new technical indicators</li>
        <li>Modify reward function</li>
        <li>Change action space (e.g., add position sizing)</li>
        <li>Adjust risk management rules</li>
      </ul>

      <h3>Training Parameters</h3>
      <p>Key parameters to tune:</p>
      <pre><code># DQN Parameters
learning_rate=0.0003
buffer_size=10000
learning_starts=1000
target_update_interval=1000

# PPO Parameters
learning_rate=0.0003
n_steps=2048
batch_size=64
n_epochs=10</code></pre>

      <h3>Risk Management</h3>
      <p>Adjust risk parameters:</p>
      <pre><code>initial_balance=10000      # Starting capital
max_position_size=0.3      # Max 30% position size
stop_loss=0.05            # 5% stop loss
take_profit=0.1           # 10% take profit
transaction_cost=0.001    # 0.1% transaction cost</code></pre>
    </div>

    <div class="section">
      <h2>📈 Performance Metrics</h2>
      <ul>
        <li><strong>Total Return</strong>: Overall portfolio performance</li>
        <li><strong>Sharpe Ratio</strong>: Risk-adjusted returns</li>
        <li><strong>Maximum Drawdown</strong>: Largest portfolio decline</li>
        <li><strong>Win Rate</strong>: Percentage of profitable trades</li>
        <li><strong>Volatility</strong>: Portfolio volatility measurement</li>
      </ul>
    </div>

    <div class="section">
      <h2>⚠️ Important Notes</h2>
      <h3>Limitations</h3>
      <ul>
        <li><strong>Not Financial Advice</strong>: This is for educational purposes only</li>
        <li><strong>Past Performance</strong>: Historical results don't guarantee future performance</li>
        <li><strong>Market Conditions</strong>: Model performance varies with market conditions</li>
        <li><strong>Transaction Costs</strong>: Real trading involves additional costs and slippage</li>
      </ul>

      <h3>Recommendations</h3>
      <ul>
        <li><strong>Paper Trading</strong>: Test thoroughly before using real money</li>
        <li><strong>Risk Management</strong>: Always use appropriate position sizing</li>
        <li><strong>Model Validation</strong>: Validate on out-of-sample data</li>
        <li><strong>Regular Retraining</strong>: Update models with new data periodically</li>
      </ul>
    </div>

    <div class="section">
      <h2>🐛 Troubleshooting</h2>
      <h3>Common Issues</h3>
      <h4>Model Loading Errors</h4>
      <pre><code># Ensure model file is correct format
if uploaded_model.name.endswith('.zip'):
    model = DQN.load(uploaded_model)</code></pre>

      <h4>Data Download Issues</h4>
      <pre><code># Check ticker symbol validity
stock = yf.Ticker(symbol)
info = stock.info
if 'regularMarketPrice' not in info:
    st.error("Invalid ticker symbol")</code></pre>

      <h4>Memory Issues</h4>
      <ul>
        <li>Reduce <code>TRAINING_TIMESTEPS</code> for faster training</li>
        <li>Use smaller datasets for testing</li>
        <li>Clear browser cache if Streamlit becomes slow</li>
      </ul>
    </div>

    <div class="section">
      <h2>📞 Support</h2>
      <ol>
        <li>Check the troubleshooting section.</li>
        <li>Review the code comments.</li>
        <li>Test with sample data first.</li>
        <li>Ensure all dependencies are installed correctly.</li>
      </ol>
    </div>

    <div class="section">
      <h2>📜 License</h2>
      <p>This project is for educational purposes. Use at your own risk.</p>
    </div>

    <div class="section">
      <h2>🔄 Updates</h2>
      <h3>Version 1.0</h3>
      <ul>
        <li>Initial release with DQN and PPO support</li>
        <li>Basic risk management features</li>
        <li>Streamlit web interface</li>
      </ul>

      <h3>Future Enhancements</h3>
      <ul>
        <li>Multi-asset portfolio support</li>
        <li>Advanced technical indicators</li>
        <li>Real-time data feeds</li>
        <li>Model ensemble methods</li>
        <li>Advanced risk metrics</li>
        <li>Mobile-responsive design</li>
      </ul>
    </div>

    <div class="section">
      <p><strong>Happy Trading! 📈🚀</strong></p>
      <p><em>Remember: This is for educational purposes only. Always do your own research and never invest more than you can afford to lose.</em></p>
    </div>
  </div>

  <script>
    function toggleTheme() {
      document.body.classList.toggle('dark');
    }
  </script>
</body>
</html>
