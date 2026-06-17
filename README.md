# NASDAQ Downside-Risk Overlay v4

This repository contains a research prototype for a NASDAQ downside-risk overlay strategy.

## What this project does

- Estimates short-horizon downside-risk events instead of predicting market direction
- Converts risk probabilities into exposure control signals
- Applies next-day execution, turnover-based transaction costs, and stress-period diagnostics
- Compares the strategy against matched-exposure and simple rule-based baselines

## Main files

- `src/quant_strategy_v4.py`: end-to-end research and backtest script
- `docs/Final_Quant_Project_Package_v4_KR.md`: main portfolio report in Korean
- `docs/Quant_Project_Technical_Report_v4_KR.md`: technical report in Korean
- `docs/Quant_Project_Glossary_v4_KR.md`: glossary in Korean
- `outputs/figures/final_backtesting_v4.png`: backtest comparison chart
- `outputs/csv/*_v4.csv`: diagnostics, sensitivity, and stress-test outputs

## Repository layout

```text
.
├── src/              # Strategy and backtest code
├── docs/             # Korean reports and glossary
├── docs/archive/     # Earlier archived report drafts
├── outputs/csv/      # Generated diagnostic tables
├── outputs/figures/  # Generated charts
├── .env.example      # Local environment template
└── requirements.txt
```

## Setup

1. Copy `.env.example` to `.env`, or create a local `.env` file in the repository root.
2. Put your FRED API key in that file:

```env
FRED_API_KEY=YOUR_FRED_API_KEY_HERE
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the strategy script:

```bash
python src/quant_strategy_v4.py
```

## Important note

- The `.env` file is gitignored and must not be uploaded.
- This project is positioned as a research prototype for a downside-risk overlay, not as a production-ready live trading system.
