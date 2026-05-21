# Technical Report: NASDAQ Downside-Risk Overlay v4

**Author:** Quant Analysis Team  
**Version:** v4  
**Use Case:** portfolio risk overlay / interview portfolio review

## 1. Research Goal
The project goal is not to maximize unconditional upside capture.  
It is to identify periods where downside risk is elevated and reduce NASDAQ exposure before drawdowns deepen.

That framing matters because it changes:
- the target definition
- the validation standard
- the interpretation of performance

## 2. v4 Hypothesis
The working hypothesis is:

1. liquidity, rates, volatility, and credit stress contain information about short-horizon downside risk
2. a calibrated risk probability is more useful than a raw class label
3. exposure control should combine regime signal and volatility scaling
4. the final overlay should beat a simple matched-exposure de-risking benchmark, not only buy-and-hold

## 3. Data and Features
The model uses:
- NASDAQ
- VIX and MOVE
- FRED liquidity and balance-sheet data
- rates, HY spread, and stress indicators
- a small set of large-cap stocks for liquidity-beta analysis

Feature treatment in v4:
- liquidity variables remain mostly in return form
- macro-stress variables are converted to rolling z-scores
- lagged features are used to preserve timing discipline

This is intended to reduce the regime-break problem from feeding raw rate/stress levels into a tree model.

## 4. Target Design
The label is a downside-risk event over a 20-trading-day horizon with a -5% threshold.

v4 uses **true forward drawdown** logic rather than a simple future minimum relative to the current close.  
This is important because the model should respond to path-dependent drawdown risk, not only endpoint weakness.

## 5. Validation Structure
### 5.1 Leakage Control
- purged time-series cross-validation
- embargo window
- scaler inside pipeline
- next-day execution timing

### 5.2 Model Training
- XGBoost as primary classifier
- logistic regression as baseline
- class-imbalance handling through `scale_pos_weight`
- inner CV kept purged instead of plain `TimeSeriesSplit`

### 5.3 Probability Handling
- out-of-fold probabilities are generated first
- calibration layer is applied
- EMA smoothing is applied fold-locally
- hysteresis and exposure mapping sit on top of calibrated probabilities

This separates the model score from the actual trading decision layer.

## 6. Trading Layer
The final trading process is:

1. estimate downside-risk probability
2. smooth the probability path
3. convert probability to signal / exposure
4. scale exposure by realized volatility
5. execute on next day
6. charge turnover-aware transaction costs

The cost model in v4 uses **target exposure turnover**, not only signal flips.

## 7. Main Results
### 7.1 Core Metrics
| Metric | Value |
| --- | ---: |
| PR-AUC | 0.3267 |
| Brier Score | 0.2508 |
| Log Loss | 1.2853 |
| Accuracy (secondary) | 0.6609 |
| Risk-Event Base Rate | 0.2875 |
| Final Strategy Return | 1.4219 |
| NASDAQ Return | 1.4966 |
| Sharpe Ratio | 1.4480 |
| Max Drawdown | -12.37% |

Accuracy is retained only as a secondary number.  
The primary read is now probability quality plus trading outcome.

### 7.2 Strategy Comparison
| Strategy | Final Return | Sharpe | MDD | Avg. Exposure | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost Risk Model | 1.4219 | 1.4480 | -12.37% | 71.70% | 6.21 |
| Continuous Overlay | 1.4219 | 1.4480 | -12.37% | 71.70% | 6.21 |
| Matched Avg Exposure | 1.3528 | 1.1542 | -17.91% | 71.54% | 0.39 |
| Logistic Risk Baseline | 1.1136 | 0.5764 | -13.87% | 42.93% | 9.87 |
| MA200 Rule Baseline | 1.2940 | 1.0447 | -12.06% | 81.06% | 9.58 |

The most important comparison is **Matched Avg Exposure**.  
That benchmark shows that v4 is not only reducing average exposure; it is also shaping exposure better than a static de-risked allocation.

### 7.3 Attribution
| Component | Final Return | Sharpe | MDD | Avg. Exposure | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buy & Hold | 1.5063 | 1.1542 | -24.32% | 99.78% | 0.55 |
| Matched Avg Exposure | 1.3528 | 1.1542 | -17.91% | 71.54% | 0.39 |
| XGBoost Signal Only | 1.5805 | 1.4062 | -17.09% | 79.74% | 1.65 |
| Vol Scaling Only | 1.3648 | 1.1307 | -17.58% | 87.99% | 7.24 |
| Final: XGBoost + Vol Scaling | 1.4219 | 1.4480 | -12.37% | 71.70% | 6.21 |

This suggests:
- signal helps select regimes
- scaling helps reduce tail shape
- combined overlay gives the strongest risk-adjusted outcome

## 8. Cost and Stress Validation
### 8.1 Cost Sensitivity
| Cost (bps) | Final Return | Sharpe | MDD |
| ---: | ---: | ---: | ---: |
| 0 | 1.4462 | 1.5154 | -12.03% |
| 5 | 1.4381 | 1.4929 | -12.15% |
| 10 | 1.4300 | 1.4705 | -12.26% |
| 15 | 1.4219 | 1.4480 | -12.37% |
| 30 | 1.3979 | 1.3804 | -12.70% |
| 50 | 1.3666 | 1.2902 | -13.14% |

The profile remains positive under higher cost assumptions, but turnover is now meaningfully higher once full exposure turnover is charged.

### 8.2 Stress Windows
| Type | Period | Start | End | Strategy Return | Market Return | Strategy MDD | Market MDD |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Predefined Stress | OOS AI / High-Rate Cycle | 2024-04-25 | 2025-12-31 | 1.4165 | 1.4253 | -10.61% | -24.32% |
| Predefined Stress | Recent OOS Window | 2025-04-21 | 2026-04-20 | 1.2839 | 1.4984 | -12.37% | -13.21% |
| Worst Window | Worst Market 63D Window | 2025-01-07 | 2025-04-08 | 0.9301 | 0.7686 | -8.31% | -23.87% |
| Worst Window | Worst Strategy 63D Window | 2025-12-29 | 2026-03-30 | 0.8898 | 0.8814 | -12.07% | -12.84% |

The stress result is strongest in the worst market 63-day window, where drawdown is substantially smaller than the market.

## 9. Sensitivity and Diagnostics
### 9.1 Target Sensitivity
The `20D / -5%` specification remains the strongest tested target among the current grid.  
Longer-horizon `40D` targets improve PR-AUC numerically but weaken portfolio results.

### 9.2 Signal Sensitivity
After introducing continuous exposure mapping, signal sensitivity is no longer perfectly flat.  
However, the variation remains moderate, which suggests the model is somewhat robust but still compressed around a limited probability range.

### 9.3 Feature Ablation
Feature-group ablation changes PR-AUC more than final PnL.  
This implies that different feature blocks alter ranking quality, but much of that difference is still absorbed by the exposure-conversion layer.

### 9.4 Risk Deciles
`risk_decile_summary_v4.csv` shows that probability ranking is not perfectly monotonic by realized event rate.  
That is a useful warning sign: calibration improved interpretability, but the probability surface is still coarse in places.

## 10. Limitations
1. post-2023 regime concentration remains a real limitation
2. PR-AUC is better than naive baselines but still modest
3. feature ablation and signal sensitivity still show partial compression
4. the final overlay remains slightly below buy-and-hold on raw return
5. liquidity-beta analysis is still more explanatory than directly executable

## 11. Practical Reading
The strongest defensible claim is not:
"this strategy beats NASDAQ."

The stronger claim is:
"this strategy produces a more disciplined downside-risk overlay than static de-risking, with Sharpe above 1 and materially lower drawdown than buy-and-hold."

## 12. Files
- `quant_strategy_v4.py`
- `final_backtesting_v4.png`
- `cost_sensitivity_v4.csv`
- `stress_analysis_v4.csv`
- `target_sensitivity_v4.csv`
- `signal_sensitivity_v4.csv`
- `signal_diagnostics_v4.csv`
- `feature_ablation_v4.csv`
- `risk_decile_summary_v4.csv`
