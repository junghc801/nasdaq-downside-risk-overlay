# Building a NASDAQ Downside-Risk Overlay Instead of Another Directional Model

Most early quant projects ask a familiar question:
can I predict whether the market goes up or down?

This project eventually moved away from that framing.

The better question turned out to be:
**when does downside risk become high enough that I should cut NASDAQ exposure?**

That shift changed almost everything about the model.

## What changed in v4
Earlier versions looked better on paper than they deserved to.
The big problems were:
- an imperfect drawdown label
- same-day timing assumptions
- weak cost realism
- too much emphasis on accuracy

v4 fixes the structure first:
- true forward drawdown labeling
- next-day execution
- purged validation
- macro-stress z-score features
- class-imbalance handling
- calibrated probabilities
- turnover-aware cost charging

The result is a model that is harder to overclaim and easier to defend.

## What the model actually does
The system uses liquidity, rates, volatility, and credit-stress information to estimate the probability of a downside event over the next 20 trading days.

That risk estimate is then translated into exposure control.
Instead of asking for full market participation all the time, the overlay tries to stay invested when risk looks manageable and reduce exposure when conditions deteriorate.

## v4 headline numbers
- PR-AUC: **0.3267**
- Brier Score: **0.2508**
- Log Loss: **1.2853**
- Final Strategy Return: **1.4219**
- NASDAQ Return: **1.4966**
- Sharpe Ratio: **1.4480**
- Max Drawdown: **-12.37%**

This is the important part:
the strategy still trails buy-and-hold on raw return, but it cuts drawdown roughly in half relative to the market.

That is exactly why the project now makes more sense as a **risk overlay** than as an alpha strategy.

## The most useful comparison
One of the better checks in v4 is the matched-exposure benchmark.

The final model and the matched benchmark carry almost the same average exposure:
- v4 model exposure: **71.70%**
- matched benchmark exposure: **71.54%**

But the final model still does better:
- model Sharpe: **1.4480**
- matched benchmark Sharpe: **1.1542**
- model MDD: **-12.37%**
- matched benchmark MDD: **-17.91%**

That matters because it suggests the result is not only "holding less NASDAQ."
It is also about *when* the model chooses to hold less.

## What I learned from the diagnostics
Two findings stood out.

First, probability quality and trading results are not the same thing.  
Feature-group ablation changed PR-AUC more than final PnL, which means the signal-conversion layer still compresses some information.

Second, once full turnover costs were charged, performance came down a bit but remained solid:
- 0 bps cost Sharpe: **1.5154**
- 15 bps cost Sharpe: **1.4480**
- 50 bps cost Sharpe: **1.2902**

That is a much more believable profile than a frictionless backtest.

## Where the model still falls short
This is not a finished production system.

The weak points are still visible:
- post-2023 regime concentration
- only modest PR-AUC
- raw return still below buy-and-hold
- probability surface still somewhat compressed

So the strongest claim is not "this beats the market."

The stronger and more honest claim is:
**this is a materially improved downside-risk overlay research package with defensible validation and a better drawdown profile than passive exposure.**

## Files worth opening
- [`final_backtesting_v4.png`](C:/Users/user/claude_test/quant_learning/quant_model_v4/final_backtesting_v4.png)
- [`cost_sensitivity_v4.csv`](C:/Users/user/claude_test/quant_learning/quant_model_v4/cost_sensitivity_v4.csv)
- [`stress_analysis_v4.csv`](C:/Users/user/claude_test/quant_learning/quant_model_v4/stress_analysis_v4.csv)
- [`target_sensitivity_v4.csv`](C:/Users/user/claude_test/quant_learning/quant_model_v4/target_sensitivity_v4.csv)
- [`signal_sensitivity_v4.csv`](C:/Users/user/claude_test/quant_learning/quant_model_v4/signal_sensitivity_v4.csv)
- [`feature_ablation_v4.csv`](C:/Users/user/claude_test/quant_learning/quant_model_v4/feature_ablation_v4.csv)

If I were presenting this in an interview, I would not lead with "I built a market-beating model."
I would lead with:
"I started with a fragile directional system, rebuilt it into a downside-risk overlay, tightened the validation logic, and ended with a more realistic strategy that still maintains Sharpe above 1 while materially reducing drawdown." 
