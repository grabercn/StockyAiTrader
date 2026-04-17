<p align="center">
  <img src="icon.png" alt="Stocky Suite" width="80" />
</p>

<h1 align="center">Stocky Suite</h1>

<p align="center">
  <strong>AI-Powered Trading Suite</strong><br/>
  <em>Scan. Analyze. Auto-Invest. All from one dashboard.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/tests-109%20passing-brightgreen?style=for-the-badge" alt="Tests" />
  <img src="https://img.shields.io/badge/AI-LightGBM%20%2B%20FinBERT-purple?style=for-the-badge" alt="AI" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/addons-10%20plugins-orange?style=flat-square" alt="Addons" />
  <img src="https://img.shields.io/badge/features-38%20ML%20signals-blue?style=flat-square" alt="Features" />
  <img src="https://img.shields.io/badge/broker-Alpaca%20Paper-yellow?style=flat-square" alt="Broker" />
</p>

---

## What is Stocky Suite?

Stocky Suite is a **free, open-source AI trading application** that combines machine learning, financial sentiment analysis, and technical indicators to help you make smarter trading decisions.

It's a **complete trading workstation** in a single app — scan dozens of stocks simultaneously, get AI-powered BUY/SELL/HOLD recommendations with confidence scores, and auto-invest with risk-managed bracket orders. All with a premium, animated UI that feels like a professional trading terminal.

**Built by [@grabercn](https://github.com/grabercn)** as a fully functional trading tool. Bring your own API keys and start analyzing.

---

## Features

### Multi-Stock AI Scanner
Scan 24+ stocks concurrently. Each ticker is analyzed through a full ML pipeline — technical indicators, sentiment analysis, and addon signals — then ranked by opportunity score.

### LightGBM + FinBERT Engine
- **LightGBM** gradient-boosted classifier trained on 38 features per ticker
- **FinBERT** financial sentiment from news headlines
- **Triple Barrier Labeling** for realistic trade outcome labels
- Time-series cross-validation prevents lookahead bias

### 10 Plug-and-Play Addons
Drop a Python file in `addons/` and it's automatically discovered. No core changes needed.

| Addon | What it provides | Needs API Key? |
|-------|-----------------|----------------|
| StockTwits Sentiment | Retail trader bullish/bearish ratio | No |
| Reddit WSB Mentions | WallStreetBets mention tracking | No |
| SPY Market Correlation | Market regime + stock-to-SPY correlation | No |
| CNN Fear & Greed Index | Market-wide sentiment (0-100) | No |
| SEC Insider Trades | Insider buying/selling from SEC filings | No |
| Twitter-RoBERTa | Social media sentiment (500MB model) | No |
| FinBERT-Tone | Analyst report sentiment (420MB model) | No |
| TimeGPT Forecast | AI price forecast via Nixtla | Yes (free tier) |
| FRED Macro Indicators | VIX, yield curve, fed rate | Yes (free) |
| Finnhub Calendar | Earnings dates + economic events | Yes (free) |

### Risk Management
- ATR-based position sizing (adapts to volatility)
- 2% max risk per trade, 5% daily drawdown limit
- Bracket orders with automatic stop-loss and take-profit
- Max 5 simultaneous positions

### 8-Tab Dashboard
| Tab | Purpose |
|-----|---------|
| **Dashboard** | Portfolio overview, positions, equity chart, activity feed |
| **Scanner** | Multi-stock scan, rank, select, auto-invest |
| **Day Trade** | Single-stock intraday analysis with chart |
| **Long Trade** | Long-term SMA/EMA crossover outlook |
| **Logs** | Every decision logged with reasoning + feature importances |
| **Tax Reports** | IRS Form 8949 / Schedule D CSV export |
| **Testing** | System diagnostics + built-in pytest runner |
| **Settings** | API keys, hardware profiles, addons, model manager |

### Premium UI
- Animated gradient headers on every panel
- Custom SVG icon library (19 icons, any size/color)
- Loading bars with shimmer sweep animation
- Floating particle effects on boot/wizard screens
- Light/dark theme auto-detection (or manual override)
- Ctrl+/- zoom scaling
- First-run setup wizard

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
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. Launch
```bash
python StockySuite.py
```

### 3. Setup
The first-run wizard walks you through:
1. **Alpaca API keys** — Get free paper trading keys at [alpaca.markets](https://alpaca.markets)
2. **Theme** — Auto-detects your Windows light/dark mode
3. **Hardware profile** — Balanced is recommended for most laptops

### 4. Trade
- Go to the **Scanner** tab
- Click **"Top 24"** preset
- Hit **SCAN & RANK**
- Select the stocks you like
- Click **AUTO-INVEST SELECTED**

---

## Hardware Profiles

| Profile | Addons | Best For |
|---------|--------|----------|
| **Max** | 10/10 | Desktop with 16GB+ RAM |
| **Balanced** | 7/10 | Most laptops (recommended) |
| **Light** | 5/10 | Older hardware, speed priority |
| **Minimal** | 0/10 | Core engine only, fastest |

---

## Project Structure

```
StockyAiTrader/
├── StockySuite.py              # Main entry — unified dashboard
├── build_exe.py                # Windows EXE builder
├── run_tests.py                # pytest runner
├── StockyApps/
│   ├── core/                   # 17 shared modules
│   │   ├── ui/                 # Custom UI framework
│   │   │   ├── animations.py   # Fade, slide, pulse, shake effects
│   │   │   ├── icons.py        # 19 SVG icons, any size/color
│   │   │   ├── backgrounds.py  # Glass panels, gradient headers
│   │   │   ├── charts.py       # Candlestick, gauge, sparkline
│   │   │   ├── tables.py       # Premium tables with auto-coloring
│   │   │   ├── theme.py        # Light/dark color provider
│   │   │   ├── boot_screen.py  # Animated loading screen
│   │   │   └── setup_wizard.py # First-run configuration
│   │   ├── model.py            # LightGBM training/prediction
│   │   ├── features.py         # 24 technical indicators
│   │   ├── risk.py             # Position sizing + drawdown limits
│   │   ├── scanner.py          # Multi-stock concurrent analysis
│   │   ├── broker.py           # Alpaca API wrapper
│   │   ├── logger.py           # JSONL decision logging
│   │   └── ...
│   └── addons/                 # 10 plug-and-play signal addons
├── tests/                      # 109+ unit + integration tests
└── .github/workflows/          # CI + auto-build releases
```

---

## API Keys You'll Need

| Service | Cost | Where to Get |
|---------|------|-------------|
| **Alpaca** (required) | Free | [alpaca.markets](https://alpaca.markets) |
| FRED | Free | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| Finnhub | Free | [finnhub.io](https://finnhub.io/register) |
| Nixtla TimeGPT | Free tier | [dashboard.nixtla.io](https://dashboard.nixtla.io/) |

Only Alpaca is required. Everything else is optional and enhances accuracy.

---

## Testing

```bash
# Run all tests
python run_tests.py

# Run specific test file
python run_tests.py -k "test_risk"

# Or use the built-in Testing tab in the app
```

109 tests covering: features, model training, risk management, labeling, signals, logging, addons, profiles, tax reports, UI framework, event bus, and full integration pipelines.

---

## Contributing

This is an open-source project. Feel free to:
- Add new addons (just drop a `.py` file in `StockyApps/addons/`)
- Improve the UI (custom widgets in `core/ui/`)
- Add more technical indicators (edit `core/features.py`)
- Build integrations for other brokers

---

## Disclaimer

This software is for **educational and paper trading purposes**. It does not constitute financial advice. Always do your own research before trading with real money. Past performance of AI models does not guarantee future results.

---

<p align="center">
  <strong>Stocky Suite</strong> — AI-Powered Trading Suite<br/>
  <em>Free & Open Source</em><br/>
  Built by <a href="https://github.com/grabercn">@grabercn</a>
</p>
