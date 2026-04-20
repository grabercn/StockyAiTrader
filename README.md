<p align="center">
  <img src="icon.png" alt="Stocky Suite" width="100" />
</p>

<h1 align="center">Stocky Suite</h1>

<p align="center">
  <strong>AI-Powered Trading Suite</strong><br/>
  <em>Scan. Analyze. Auto-Invest. All from one dashboard.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.0-0ea5e9?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBvbHlsaW5lIHBvaW50cz0iMjIsNiAxMy41LDE0LjUgOC41LDkuNSAyLDE2IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiLz48L3N2Zz4=&logoColor=white" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/tests-174%20passing-10b981?style=for-the-badge" alt="Tests" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-LightGBM%20%2B%20FinBERT%20%2B%20GPT--2-6366f1?style=flat-square" alt="AI" />
  <img src="https://img.shields.io/badge/addons-10%20plugins-f59e0b?style=flat-square" alt="Addons" />
  <img src="https://img.shields.io/badge/features-38%20ML%20signals-0ea5e9?style=flat-square" alt="Features" />
  <img src="https://img.shields.io/badge/broker-Alpaca-10b981?style=flat-square" alt="Broker" />
  <img src="https://img.shields.io/badge/sectors-15-94a3b8?style=flat-square" alt="Sectors" />
</p>

---

## What is Stocky Suite?

Stocky Suite is a **free, open-source AI trading application** that combines machine learning, financial sentiment analysis, and technical indicators to help you make smarter trading decisions.

It's a **complete trading workstation** in a single app — scan hundreds of stocks simultaneously, get AI-powered BUY/SELL/HOLD recommendations with confidence scores, and auto-invest with risk-managed bracket orders.

**Built by [@grabercn](https://github.com/grabercn)**. Bring your own API keys and start analyzing.

---

## Key Features

### Intelligent Multi-Stock Scanner
- Scan **500+ stocks** from the S&P 500, trending lists, or any custom tickers
- **Smart mode** auto-picks optimal training period + bar interval per stock based on volatility
- Live data sources: Most Active, Day Gainers/Losers, Trending Social, High Volume, 15 sectors
- Per-stock auto-trade monitoring with adaptive check intervals

### AI Engine
- **LightGBM** gradient-boosted classifier trained on **38 features** per ticker
- **FinBERT** financial-domain sentiment from news headlines
- **DistilGPT-2** generates natural-language trade reasoning
- **Triple Barrier Labeling** for realistic trade outcome labels
- Time-series cross-validation prevents lookahead bias
- **Smart position sizing** based on ATR, buying power, confidence, and diversification

### 10 Plug-and-Play Addons

| Addon | What it provides | Needs Key? |
|-------|-----------------|:----------:|
| StockTwits Sentiment | Retail trader bullish/bearish ratio | |
| Reddit WSB Mentions | WallStreetBets mention tracking | |
| SPY Market Correlation | Market regime + correlation | |
| CNN Fear & Greed | Market-wide sentiment (0-100) | |
| SEC Insider Trades | Insider buying/selling from SEC | |
| Twitter-RoBERTa | Social media sentiment model | |
| FinBERT-Tone | Analyst report sentiment model | |
| TimeGPT Forecast | AI price forecast via Nixtla | Yes |
| FRED Macro | VIX, yield curve, fed rate | Yes |
| Finnhub Calendar | Earnings dates + economic events | Yes |

### Risk Management
- ATR-based position sizing adapted to volatility, confidence, and portfolio state
- Bracket orders with automatic stop-loss and take-profit
- Daily drawdown limit (5%) and position concentration cap (10%)
- **4 aggressivity profiles**: Chill, Default, Aggressive, YOLO

### 9-Tab Dashboard

| Tab | Purpose |
|-----|---------|
| **Dashboard** | Portfolio overview, equity chart, positions with sell buttons, active orders, activity feed |
| **Scanner** | Live stock discovery, AI scan, auto-trade toggle, detail panel with Deep Analyze |
| **Portfolio** | Holdings, trade history with costs, watchlists, performance chart |
| **Day Trade** | Single-stock intraday analysis with chart + hover tooltips |
| **Long Trade** | Long-term SMA/EMA crossover outlook |
| **Logs** | Every decision logged with reasoning, feature importances, filters |
| **Tax Reports** | IRS Form 8949 / Schedule D CSV export |
| **Testing** | System diagnostics + built-in pytest runner (174 tests) |
| **Settings** | API keys, hardware profiles, aggressivity, theme, zoom, addons, models |

### Premium UI
- Animated gradient headers on every panel
- 25+ custom SVG icons, rendered at 2x for crisp high-DPI
- Animated loading bars with shimmer sweep effect
- Boot screen with floating color orbs
- First-run setup wizard
- Light/dark theme auto-detection
- Notification bell with history overlay and type filters
- System tray — app keeps trading when minimized
- Windows toast notifications on trade execution
- Auto-update checker from GitHub releases

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/grabercn/StockyAiTrader.git
cd StockyAiTrader
pip install -r requirements.txt
```

> **Windows + no GPU?** Install CPU-only PyTorch:
> ```bash
> pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. Launch
```bash
python StockySuite.py
```

### 3. Setup
The first-run wizard walks you through:
1. **Alpaca API keys** — free paper trading at [alpaca.markets](https://alpaca.markets)
2. **Theme** — auto-detects your Windows light/dark mode
3. **Hardware profile** — Balanced recommended for most laptops

### 4. Trade
- Go to **Scanner** tab
- Select **"Most Active Today"** from the source dropdown
- Hit **SCAN & RANK**
- Click any stock for detailed analysis
- Click the robot icon to enable auto-trading

---

## Profiles

### Hardware Profiles
| Profile | Addons | Best For |
|---------|--------|----------|
| **Max** | 10/10 | Desktop with 16GB+ RAM |
| **Balanced** | 7/10 | Most laptops (recommended) |
| **Light** | 5/10 | Older hardware |
| **Minimal** | 0/10 | Core engine only |

### Trading Aggressivity
| Profile | Confidence | Position Size | Max Trades/Day |
|---------|-----------|--------------|----------------|
| **Chill** | 70% | 0.5x | 3 |
| **Default** | 50% | 1.0x | 8 |
| **Aggressive** | 35% | 1.5x | 15 |
| **YOLO** | 25% | 2.0x | 30 |

---

## Project Structure

```
StockyAiTrader/
├── StockySuite.py              # Main entry (524 lines)
├── StockyApps/
│   ├── panels/                 # 12 modular panel files
│   │   ├── dashboard.py        # Portfolio overview
│   │   ├── scanner.py          # Multi-stock scanner
│   │   ├── portfolio.py        # Holdings + trade history
│   │   ├── day_trade.py        # Intraday analysis
│   │   ├── long_trade.py       # Long-term outlook
│   │   ├── settings_panel.py   # Configuration
│   │   └── ...
│   ├── core/                   # 20+ shared modules
│   │   ├── auto_trader.py      # Background trading service
│   │   ├── intelligent_trader.py # Adaptive interval engine
│   │   ├── llm_reasoner.py     # GPT-2 trade reasoning
│   │   ├── order_manager.py    # Alpaca order lifecycle
│   │   ├── scanner.py          # Smart per-stock analysis
│   │   ├── discovery.py        # Live ticker discovery
│   │   ├── risk.py             # Smart position sizing
│   │   ├── updater.py          # Auto-update checker
│   │   ├── tray_agent.py       # System tray + notifications
│   │   ├── ui/                 # Custom UI framework
│   │   └── ...
│   └── addons/                 # 10 plug-and-play signal addons
├── tests/                      # 174 unit + integration tests
└── .github/workflows/          # CI + auto-build releases
```

---

## API Keys

| Service | Cost | Required? | Link |
|---------|------|:---------:|------|
| **Alpaca** | Free | Yes | [alpaca.markets](https://alpaca.markets) |
| FRED | Free | No | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| Finnhub | Free | No | [finnhub.io](https://finnhub.io/register) |
| Nixtla TimeGPT | Free tier | No | [dashboard.nixtla.io](https://dashboard.nixtla.io/) |

---

## Testing

```bash
python run_tests.py          # Run all 174 tests
python run_tests.py -k risk  # Run specific tests
```

Or use the **Testing** tab in the app for diagnostics + one-click test runner.

---

## Contributing

- Add addons: drop a `.py` file in `StockyApps/addons/`
- Add indicators: edit `core/features.py`
- Add panels: create a file in `StockyApps/panels/`
- Add brokers: extend `core/broker.py`

---

## Disclaimer

This software is for **educational and paper trading purposes**. It does not constitute financial advice. Always do your own research before trading with real money. Past performance of AI models does not guarantee future results.

---

<p align="center">
  <img src="icon.png" width="32" /><br/>
  <strong>Stocky Suite v5.0</strong><br/>
  <em>AI-Powered Trading Suite — Free & Open Source</em><br/>
  Built by <a href="https://github.com/grabercn">@grabercn</a>
</p>
