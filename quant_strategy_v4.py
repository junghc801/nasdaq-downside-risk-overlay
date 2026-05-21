import pandas as pd
import numpy as np
import yfinance as yf
import os
from fredapi import Fred
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

# ==========================================
# [환경 설정]
# ==========================================
ENV_PATH = Path(__file__).resolve().parent / '.env'
load_dotenv(ENV_PATH)

FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError(
        f"FRED_API_KEY is missing. Add it to {ENV_PATH} before running this script."
    )

fred = Fred(api_key=FRED_API_KEY)
start_date = '2019-01-01'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
OUTPUT_DIR = Path(__file__).resolve().parent

# 핵심 종목 리스트 (유동성 베타 분석용)
core_stocks = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'TSLA', 'META']

print("1. 고수준 데이터 수집 및 공시 시차 보정 중...")

# 1-1. FRED 데이터 (주간/일간)
# WALCL, WRESBAL, WTREGEN은 주간(수요일 기준) 데이터이며 목요일 오후 발표됨.
# 실무적으로 금요일부터 매매에 활용 가능하므로 2일의 Lag를 적용하여 Look-ahead Bias 방지.
# DGS10, T10Y2Y, BAMLH0A0HYM2, STLFSI4, DFF는 일간 데이터임.
fred_series = {
    'Fed_Total_Assets': 'WALCL',
    'Reserves': 'WRESBAL',
    'TGA': 'WTREGEN',
    'RRP': 'RRPONTSYD',
    '10Y_Treasury': 'DGS10',
    'Yield_Curve': 'T10Y2Y',
    'HY_Spread': 'BAMLH0A0HYM2',
    'Financial_Stress': 'STLFSI4',
    'Fed_Funds_Rate': 'DFF'
}

df_fred = pd.DataFrame()
for name, series_id in fred_series.items():
    s = fred.get_series(series_id, observation_start=start_date)
    # 주간 데이터(WALCL, WRESBAL, TGA) 식별 및 시차 보정
    if series_id in ['WALCL', 'WRESBAL', 'WTREGEN']:
        # 수요일 기준 데이터를 금요일로 밀어 발표 시점 반영
        s.index = s.index + pd.Timedelta(days=2)
    df_fred[name] = s

# 1-2. 시장 데이터 (나스닥, VIX, MOVE, 핵심 종목)
# yfinance MultiIndex 해결을 위해 평탄화 로직 포함
tickers = ['^IXIC', '^VIX', '^MOVE'] + core_stocks
df_market_raw = yf.download(tickers, start=start_date, end=end_date, progress=False)

# MultiIndex 평탄화 및 종가 추출
if isinstance(df_market_raw.columns, pd.MultiIndex):
    df_market = df_market_raw['Close'].copy()
else:
    df_market = df_market_raw.copy()

# 컬럼명 정리
df_market = df_market.rename(columns={'^IXIC': 'NASDAQ', '^VIX': 'VIX', '^MOVE': 'MOVE'})

# ==========================================
# [2단계] 데이터 통합 및 정제 (Regime Filter 준비)
# ==========================================
print("2. 데이터 통합 및 실무형 피처 엔지니어링...")

# 나스닥 날짜 기준 통합
df = pd.DataFrame(index=df_market.index)
df = df.join(df_fred).join(df_market)

# 결측치 처리 (Forward Fill)
df = df.ffill()

# 순유동성(Net Liquidity) 계산
df['Net_Liquidity'] = df['Fed_Total_Assets'] - (df['RRP'] + df['TGA'])

# 기술적 지표 추가 (Regime Filter 강화)
df['MA200'] = df['NASDAQ'].rolling(window=200).mean()
df['VIX_MA20'] = df['VIX'].rolling(window=20).mean()
df['Volatility'] = df['NASDAQ'].pct_change().rolling(window=20).std()

def rolling_zscore(series, window=252, min_periods=126):
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)

# 수익률 변환 (정상성 확보)
# 유동성 지표와 주가는 수익률(pct_change)로, 금리와 VIX, MOVE 등은 수치 자체나 차분 활용
df_model = pd.DataFrame(index=df.index)
df_model['NASDAQ_Ret'] = df['NASDAQ'].pct_change()
df_model['Reserves_Ret'] = df['Reserves'].pct_change()
df_model['Net_Liq_Ret'] = df['Net_Liquidity'].pct_change()
df_model['RRP_Ret'] = df['RRP'].pct_change()
df_model['VIX_Z'] = rolling_zscore(df['VIX'])
df_model['MOVE_Z'] = rolling_zscore(df['MOVE'])
df_model['HY_Spread_Z'] = rolling_zscore(df['HY_Spread'])
df_model['Financial_Stress_Z'] = rolling_zscore(df['Financial_Stress'])
df_model['Fed_Funds_Rate_Z'] = rolling_zscore(df['Fed_Funds_Rate'])
df_model['10Y_Treasury_Z'] = rolling_zscore(df['10Y_Treasury'])
df_model['Trend'] = (df['NASDAQ'] > df['MA200']).astype(int)

# 시차 변수 생성
# 유동성 지표 시차 (Lasso 유효 시차)
for col in ['Reserves_Ret', 'Net_Liq_Ret', 'RRP_Ret']:
    for lag in [10, 40, 60]:
        df_model[f'{col}_lag{lag}'] = df_model[col].shift(lag)

# 고빈도 지표 시차 (빠른 신호 대응용)
for col in ['MOVE_Z', 'HY_Spread_Z', 'VIX_Z']:
    for lag in [1, 5, 10]:
        df_model[f'{col}_lag{lag}'] = df_model[col].shift(lag)

# 타겟 설정: 주간 리밸런싱을 고려한 5일 뒤 방향성
TARGET_HORIZON = 20
DRAWDOWN_THRESHOLD = -0.05

def compute_true_forward_drawdown(prices, horizon):
    future_path = pd.concat(
        [prices.shift(-day) for day in range(0, horizon + 1)],
        axis=1
    )
    running_peak = future_path.cummax(axis=1)
    true_drawdown = (future_path / running_peak - 1).min(axis=1)
    true_drawdown[future_path.isna().any(axis=1)] = np.nan
    return true_drawdown

df_model['Forward_Drawdown'] = compute_true_forward_drawdown(df['NASDAQ'], TARGET_HORIZON)
df_model['Target'] = (df_model['Forward_Drawdown'] <= DRAWDOWN_THRESHOLD).astype(float)

# 무한대 및 결측치 제거
df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

# ==========================================
# [3단계] 검증 고도화 및 하이퍼파라미터 튜닝
# ==========================================
print("3. 고도화된 검증(Purging & Embargo) 및 최적 파라미터 튜닝 중...")

EMBARGO_DAYS = 5
PURGE_GAP = TARGET_HORIZON + EMBARGO_DAYS

def time_series_split_purged(X, y, n_splits=5, purge_gap=PURGE_GAP):
    """
    시계열 데이터 누수를 막기 위한 Purging이 적용된 분할기
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        test_start = test_idx[0]
        # Purging: 테스트셋 직전 5일(타겟 기간) 데이터 제거
        # Purge the target horizon plus an embargo before each test fold.
        purged_train_idx = train_idx[train_idx < (test_start - purge_gap)]
        yield purged_train_idx, test_idx

def build_purged_splits(X, y, max_splits=3, purge_gap=PURGE_GAP):
    candidate_gaps = sorted({purge_gap, max(1, purge_gap // 2), TARGET_HORIZON, EMBARGO_DAYS, 1}, reverse=True)
    for gap in candidate_gaps:
        for n_splits in range(max_splits, 1, -1):
            splits = list(time_series_split_purged(X, y, n_splits=n_splits, purge_gap=gap))
            if splits and all(len(train_idx) > 0 and len(test_idx) > 0 for train_idx, test_idx in splits):
                return splits
    raise ValueError("Not enough data to build purged CV splits.")

def apply_hysteresis_signal(prob_series, lower=0.45, upper=0.55):
    signal = pd.Series(np.nan, index=prob_series.index)
    signal.loc[prob_series < lower] = 1
    signal.loc[prob_series > upper] = 0
    return signal.ffill().fillna(0)

def map_prob_to_exposure(prob_series, lower=0.45, upper=0.55):
    exposure = pd.Series(index=prob_series.index, dtype=float)
    exposure.loc[prob_series <= lower] = 1.0
    exposure.loc[prob_series >= upper] = 0.0
    mid_mask = (prob_series > lower) & (prob_series < upper)
    exposure.loc[mid_mask] = 1.0 - ((prob_series.loc[mid_mask] - lower) / (upper - lower))
    return exposure.ffill().fillna(0.0).clip(lower=0.0, upper=1.0)

def split_train_and_calibration(X_train_fold, y_train_fold, calibration_fraction=0.2, min_calibration_size=40):
    calibration_size = max(min_calibration_size, int(len(X_train_fold) * calibration_fraction))
    calibration_size = min(calibration_size, len(X_train_fold) - 1)
    if calibration_size <= 0:
        return X_train_fold, y_train_fold, None, None
    split_point = len(X_train_fold) - calibration_size
    X_model_train = X_train_fold.iloc[:split_point]
    y_model_train = y_train_fold.iloc[:split_point]
    X_calibration = X_train_fold.iloc[split_point:]
    y_calibration = y_train_fold.iloc[split_point:]
    if len(X_model_train) == 0 or len(X_calibration) == 0:
        return X_train_fold, y_train_fold, None, None
    return X_model_train, y_model_train, X_calibration, y_calibration

def calibrate_probabilities(train_probs, train_labels, test_probs):
    if train_probs is None or train_labels is None:
        return test_probs
    if len(train_labels) == 0 or train_labels.nunique() < 2:
        return test_probs
    calibration_model = LogisticRegression(random_state=42, max_iter=1000)
    calibration_model.fit(train_probs.reshape(-1, 1), train_labels)
    return calibration_model.predict_proba(test_probs.reshape(-1, 1))[:, 1]

def safe_average_precision_scorer(estimator, X_test, y_test):
    if len(np.unique(y_test)) < 2:
        return 0.0
    y_prob = estimator.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_prob)

# 특성(X)과 정답(y) 분리
X = df_model.drop(columns=['Target', 'Forward_Drawdown'])
y = df_model['Target'].astype(int)


# 하이퍼파라미터 튜닝 (범위 축소하여 속도 확보)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__max_depth': [3, 5],
    'model__n_estimators': [50, 100],
    'model__learning_rate': [0.01, 0.05]
}

# Purged CV를 사용한 Grid Search
best_params_list = []
y_probs_raw_final = np.full(len(y), np.nan)
y_probs_final = np.full(len(y), np.nan)
y_probs_final_ema = np.full(len(y), np.nan)
y_signal_final = np.full(len(y), np.nan)
y_probs_logit = np.full(len(y), np.nan)
y_probs_logit_ema = np.full(len(y), np.nan)
y_signal_logit = np.full(len(y), np.nan)

# 실제 운용 환경과 유사하게 각 폴드마다 튜닝 후 예측
for train_idx, test_idx in time_series_split_purged(X, y):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    if y_train_fold.nunique() < 2:
        constant_prob = float(y_train_fold.iloc[0]) if len(y_train_fold) > 0 else 0.0
        constant_probs = np.full(len(test_idx), constant_prob)
        constant_series = pd.Series(constant_probs, index=X.iloc[test_idx].index)
        constant_ema = constant_series.ewm(span=5).mean()
        constant_signal = apply_hysteresis_signal(constant_ema)
        best_params_list.append({'learning_rate': np.nan, 'max_depth': np.nan, 'n_estimators': np.nan})
        y_probs_final[test_idx] = constant_probs
        y_probs_final_ema[test_idx] = constant_ema.to_numpy()
        y_signal_final[test_idx] = constant_signal.to_numpy()
        y_probs_logit[test_idx] = constant_probs
        y_probs_logit_ema[test_idx] = constant_ema.to_numpy()
        y_signal_logit[test_idx] = constant_signal.to_numpy()
        continue

    X_model_train, y_model_train, X_calibration, y_calibration = split_train_and_calibration(X_train_fold, y_train_fold)
    if X_calibration is None or y_calibration is None or y_model_train.nunique() < 2:
        X_model_train, y_model_train = X_train_fold, y_train_fold
        X_calibration, y_calibration = None, None
    
    # 내부 교차 검증 (단순화)
    inner_cv = build_purged_splits(X_model_train, y_model_train, max_splits=3, purge_gap=PURGE_GAP)
    positive_count = y_model_train.sum()
    negative_count = len(y_model_train) - positive_count
    scale_pos_weight = 1.0 if positive_count == 0 else negative_count / positive_count
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ))
    ])
    grid = GridSearchCV(model_pipeline, 
                        param_grid, cv=inner_cv, scoring=safe_average_precision_scorer, n_jobs=-1)
    grid.fit(X_model_train, y_model_train)
    
    best_model = grid.best_estimator_
    best_params_list.append({k.replace('model__', ''): v for k, v in grid.best_params_.items()})
    
    # 확률값(Probability) 추출
    xgb_raw_probs = best_model.predict_proba(X.iloc[test_idx])[:, 1]
    if X_calibration is not None and y_calibration is not None:
        calibration_probs = best_model.predict_proba(X_calibration)[:, 1]
        xgb_probs = calibrate_probabilities(calibration_probs, y_calibration, xgb_raw_probs)
    else:
        xgb_probs = xgb_raw_probs
    xgb_prob_series = pd.Series(xgb_probs, index=X.iloc[test_idx].index)
    xgb_prob_ema = xgb_prob_series.ewm(span=5).mean()
    xgb_signal = apply_hysteresis_signal(xgb_prob_ema)
    y_probs_raw_final[test_idx] = xgb_raw_probs
    y_probs_final[test_idx] = xgb_probs
    y_probs_final_ema[test_idx] = xgb_prob_ema.to_numpy()
    y_signal_final[test_idx] = xgb_signal.to_numpy()

    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    baseline_pipeline.fit(X_model_train, y_model_train)
    logit_probs = baseline_pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    logit_prob_series = pd.Series(logit_probs, index=X.iloc[test_idx].index)
    logit_prob_ema = logit_prob_series.ewm(span=5).mean()
    logit_signal = apply_hysteresis_signal(logit_prob_ema)
    y_probs_logit[test_idx] = logit_probs
    y_probs_logit_ema[test_idx] = logit_prob_ema.to_numpy()
    y_signal_logit[test_idx] = logit_signal.to_numpy()

# 신호 평활화 (EMA) 및 임계값 적용
# 예측 확률에 5일 EMA 적용하여 잦은 신호 바뀜 방지
df_probs = pd.DataFrame({'Risk_Prob': y_probs_final}, index=y.index)
df_probs['Risk_Prob_Raw'] = y_probs_raw_final
df_probs['Risk_Prob_EMA'] = y_probs_final_ema
df_probs['Logit_Risk_Prob'] = y_probs_logit
df_probs['Logit_Risk_Prob_EMA'] = y_probs_logit_ema

# 확률이 55% 이상일 때 '상승 레짐', 45% 이하일 때 '하락 레짐'으로 보수적 판단
df_probs['Signal'] = y_signal_final
df_probs['Logit_Signal'] = y_signal_logit
df_probs['Exposure'] = map_prob_to_exposure(df_probs['Risk_Prob_EMA'])
df_probs['Logit_Exposure'] = map_prob_to_exposure(df_probs['Logit_Risk_Prob_EMA'])
df_probs['MA200_Signal'] = df_model['Trend'].loc[df_probs.index].astype(float)

# 결과 추출
valid_prob_mask = df_probs['Risk_Prob_EMA'].notna()
test_start_idx = np.argmax(valid_prob_mask.to_numpy())
test_actual = y.loc[valid_prob_mask]
test_pred = (df_probs.loc[valid_prob_mask, 'Risk_Prob_EMA'] > 0.5).astype(int)
xgb_eval_probs = df_probs.loc[valid_prob_mask, 'Risk_Prob']
xgb_pr_auc = average_precision_score(test_actual, xgb_eval_probs)
xgb_brier = brier_score_loss(test_actual, xgb_eval_probs)
xgb_log_loss = log_loss(test_actual, xgb_eval_probs.clip(1e-6, 1 - 1e-6))
logit_valid_mask = df_probs['Logit_Risk_Prob_EMA'].notna()
logit_actual = y.loc[logit_valid_mask]
logit_pred = (df_probs.loc[logit_valid_mask, 'Logit_Risk_Prob_EMA'] > 0.5).astype(int)

print(f"PR-AUC: {xgb_pr_auc:.4f}")
print(f"Brier Score: {xgb_brier:.4f}")
print(f"Log Loss: {xgb_log_loss:.4f}")

decile_df = pd.DataFrame({
    'Risk_Prob': xgb_eval_probs,
    'Target': test_actual,
    'Forward_Drawdown': df_model.loc[test_actual.index, 'Forward_Drawdown']
}).dropna()
decile_df['Risk_Decile'] = pd.qcut(
    decile_df['Risk_Prob'].rank(method='first'),
    10,
    labels=False
) + 1
risk_decile_summary = decile_df.groupby('Risk_Decile').agg(
    Avg_Predicted_Prob=('Risk_Prob', 'mean'),
    Actual_Event_Rate=('Target', 'mean'),
    Avg_Forward_Drawdown=('Forward_Drawdown', 'mean'),
    Count=('Target', 'size')
).reset_index()
risk_decile_summary.to_csv(OUTPUT_DIR / 'risk_decile_summary_v4.csv', index=False)

print(f"최적 파라미터(마지막 폴드): {best_params_list[-1]}")
print(f"정확도 (Purged Accuracy): {accuracy_score(test_actual, test_pred):.4f}")

# ==========================================
# [4단계] 핵심 종목 추출: 유동성 베타 (Liquidity Beta)
# ==========================================
print("\n4. 핵심 종목별 유동성 민감도(Liquidity Beta) 분석 중...")

# 유동성 장세(Net_Liquidity 증가) 시 어떤 종목이 가장 탄력적인가?
liq_beta = {}
for stock in core_stocks:
    stock_ret = df[stock].pct_change().dropna()
    net_liq_ret = df['Net_Liquidity'].pct_change().dropna()
    
    # 공통 날짜 정렬
    common_idx = stock_ret.index.intersection(net_liq_ret.index)
    
    # 회귀 분석을 통한 베타 산출 (공식: cov(stock, liq) / var(liq))
    covariance = np.cov(stock_ret.loc[common_idx], net_liq_ret.loc[common_idx])[0, 1]
    variance = np.var(net_liq_ret.loc[common_idx])
    liq_beta[stock] = covariance / variance

df_beta = pd.Series(liq_beta).sort_values(ascending=False)
print("--- 종목별 유동성 베타 (Top List) ---")
print(df_beta)

# ==========================================
# [5단계] 실전 백테스팅 및 리스크 관리 (Volatility Scaling)
# ==========================================
print("\n5. 실전 백테스팅 (Volatility Scaling 및 수수료 반영) 수행 중...")

# 테스트 셋 구간 추출 및 신호 결합
results_df = pd.DataFrame(index=y.index[test_start_idx + 1:])
results_df['Market_Return'] = df['NASDAQ'].pct_change().loc[results_df.index]
results_df['Target_Signal'] = df_probs['Signal'].loc[results_df.index]

# Volatility Scaling (변동성 조절)
# 목표 변동성(Target Vol)을 15%로 설정하고, 현재 변동성에 따라 투자 비중 조절
target_vol = 0.15
# 연율화된 20일 이동 변동성
results_df['Current_Vol'] = df['NASDAQ'].pct_change().rolling(window=20).std().loc[results_df.index] * np.sqrt(252)
results_df['Target_Position_Size'] = (target_vol / results_df['Current_Vol']).fillna(1.0)
# 비중이 너무 커지지 않도록 최대 1.2배(레버리지 20%)로 제한
results_df['Target_Position_Size'] = results_df['Target_Position_Size'].clip(upper=1.2)

# 최종 전략 수익률 (신호 * 비중 * 시장수익률)
results_df['Executed_Signal'] = df_probs['Signal'].shift(1).loc[results_df.index].fillna(0)
results_df['Executed_Position_Size'] = results_df['Target_Position_Size'].shift(1).fillna(0)
results_df['Target_Total_Exposure'] = results_df['Target_Signal'] * results_df['Target_Position_Size']
results_df['Executed_Total_Exposure'] = results_df['Target_Total_Exposure'].shift(1).fillna(0)
results_df['Trade'] = results_df['Target_Total_Exposure'].diff().abs().fillna(results_df['Target_Total_Exposure'].abs())
transaction_cost = 0.0015

# 전략 수익률 계산
results_df['Raw_Strategy_Return'] = results_df['Executed_Total_Exposure'] * results_df['Market_Return']
results_df['Strategy_Return'] = results_df['Raw_Strategy_Return'] - (results_df['Trade'] * transaction_cost)

# 누적 수익률 및 리스크 지표
results_df['Cum_Market'] = (1 + results_df['Market_Return']).cumprod()
results_df['Cum_Strategy'] = (1 + results_df['Strategy_Return'].fillna(0)).cumprod()

# Sharpe Ratio
sharpe = np.sqrt(252) * (results_df['Strategy_Return'].mean() / results_df['Strategy_Return'].std())
# MDD
peak = results_df['Cum_Strategy'].expanding().max()
drawdown = (results_df['Cum_Strategy'] - peak) / peak
mdd = drawdown.min()

def calculate_return_metrics(returns, exposure=None, trades=None):
    returns = returns.fillna(0)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    std = returns.std()
    sharpe_value = np.nan if std == 0 or np.isnan(std) else np.sqrt(252) * (returns.mean() / std)
    metrics = {
        'Final_Return': cumulative.iloc[-1],
        'Sharpe': sharpe_value,
        'MDD': drawdown.min()
    }
    if exposure is not None:
        metrics['Avg_Exposure'] = exposure.reindex(returns.index).fillna(0).mean()
    if trades is not None:
        metrics['Turnover'] = trades.reindex(returns.index).fillna(0).sum() / max(len(returns), 1) * 252
    return metrics

def run_cv_pipeline(X_input, y_input, model_params, ema_span=5, lower=0.45, upper=0.55):
    probs = np.full(len(y_input), np.nan)
    probs_ema = np.full(len(y_input), np.nan)
    signals = np.full(len(y_input), np.nan)
    exposures = np.full(len(y_input), np.nan)

    for train_idx, test_idx in time_series_split_purged(X_input, y_input):
        X_train_fold, y_train_fold = X_input.iloc[train_idx], y_input.iloc[train_idx]
        if y_train_fold.nunique() < 2:
            constant_prob = float(y_train_fold.iloc[0]) if len(y_train_fold) > 0 else 0.0
            constant_probs = np.full(len(test_idx), constant_prob)
            constant_series = pd.Series(constant_probs, index=X_input.iloc[test_idx].index)
            constant_ema = constant_series.ewm(span=ema_span).mean()
            constant_signal = apply_hysteresis_signal(constant_ema, lower=lower, upper=upper)
            constant_exposure = map_prob_to_exposure(constant_ema, lower=lower, upper=upper)
            probs[test_idx] = constant_probs
            probs_ema[test_idx] = constant_ema.to_numpy()
            signals[test_idx] = constant_signal.to_numpy()
            exposures[test_idx] = constant_exposure.to_numpy()
            continue

        X_model_train, y_model_train, X_calibration, y_calibration = split_train_and_calibration(X_train_fold, y_train_fold)
        if X_calibration is None or y_calibration is None or y_model_train.nunique() < 2:
            X_model_train, y_model_train = X_train_fold, y_train_fold
            X_calibration, y_calibration = None, None

        positive_count = y_model_train.sum()
        negative_count = len(y_model_train) - positive_count
        scale_pos_weight = 1.0 if positive_count == 0 else negative_count / positive_count
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                max_depth=model_params['max_depth'],
                n_estimators=model_params['n_estimators'],
                learning_rate=model_params['learning_rate']
            ))
        ])
        model.fit(X_model_train, y_model_train)

        raw_probs = model.predict_proba(X_input.iloc[test_idx])[:, 1]
        if X_calibration is not None and y_calibration is not None:
            calibration_probs = model.predict_proba(X_calibration)[:, 1]
            calibrated_probs = calibrate_probabilities(calibration_probs, y_calibration, raw_probs)
        else:
            calibrated_probs = raw_probs

        prob_series = pd.Series(calibrated_probs, index=X_input.iloc[test_idx].index)
        prob_ema = prob_series.ewm(span=ema_span).mean()
        signal = apply_hysteresis_signal(prob_ema, lower=lower, upper=upper)
        exposure = map_prob_to_exposure(prob_ema, lower=lower, upper=upper)
        probs[test_idx] = calibrated_probs
        probs_ema[test_idx] = prob_ema.to_numpy()
        signals[test_idx] = signal.to_numpy()
        exposures[test_idx] = exposure.to_numpy()

    out = pd.DataFrame(index=y_input.index)
    out['Risk_Prob'] = probs
    out['Risk_Prob_EMA'] = probs_ema
    out['Signal'] = signals
    out['Exposure'] = exposures
    return out

def backtest_signal_on_index(signal_series, index, use_vol_scaling=True, apply_cost=True, constant_exposure=None):
    out = pd.DataFrame(index=index)
    out['Market_Return'] = df['NASDAQ'].pct_change().loc[index]
    out['Current_Vol'] = df['NASDAQ'].pct_change().rolling(window=20).std().loc[index] * np.sqrt(252)
    target_position_size = (target_vol / out['Current_Vol']).fillna(1.0).clip(upper=1.2)
    if constant_exposure is not None:
        out['Target_Base_Exposure'] = float(constant_exposure)
    else:
        out['Target_Base_Exposure'] = signal_series.loc[index].fillna(0)
    if use_vol_scaling:
        out['Target_Position_Size'] = target_position_size
    else:
        out['Target_Position_Size'] = 1.0
    out['Target_Total_Exposure'] = out['Target_Base_Exposure'] * out['Target_Position_Size']
    out['Executed_Signal'] = out['Target_Base_Exposure'].shift(1).fillna(0)
    out['Executed_Position_Size'] = out['Target_Position_Size'].shift(1).fillna(0)
    out['Executed_Total_Exposure'] = out['Target_Total_Exposure'].shift(1).fillna(0)
    out['Trade'] = out['Target_Total_Exposure'].diff().abs().fillna(out['Target_Total_Exposure'].abs())
    cost = transaction_cost if apply_cost else 0.0
    out['Strategy_Return'] = out['Executed_Total_Exposure'] * out['Market_Return'] - (out['Trade'] * cost)
    metrics = calculate_return_metrics(
        out['Strategy_Return'],
        exposure=out['Executed_Total_Exposure'],
        trades=out['Trade']
    )
    return out, metrics

def backtest_baseline_signal(signal, label):
    out = pd.DataFrame(index=results_df.index)
    out['Market_Return'] = results_df['Market_Return']
    out['Target_Signal'] = signal.loc[out.index].fillna(0)
    out['Current_Vol'] = results_df['Current_Vol']
    out['Target_Position_Size'] = results_df['Target_Position_Size']
    out['Target_Total_Exposure'] = out['Target_Signal'] * out['Target_Position_Size']
    out['Executed_Signal'] = signal.shift(1).loc[out.index].fillna(0)
    out['Executed_Position_Size'] = out['Target_Position_Size'].shift(1).fillna(0)
    out['Executed_Total_Exposure'] = out['Target_Total_Exposure'].shift(1).fillna(0)
    out['Trade'] = out['Target_Total_Exposure'].diff().abs().fillna(out['Target_Total_Exposure'].abs())
    out['Raw_Strategy_Return'] = out['Executed_Total_Exposure'] * out['Market_Return']
    out['Strategy_Return'] = out['Raw_Strategy_Return'] - (out['Trade'] * transaction_cost)
    out['Cum_Strategy'] = (1 + out['Strategy_Return'].fillna(0)).cumprod()
    out_peak = out['Cum_Strategy'].expanding().max()
    out_drawdown = (out['Cum_Strategy'] - out_peak) / out_peak
    out_sharpe = np.sqrt(252) * (out['Strategy_Return'].mean() / out['Strategy_Return'].std())
    metrics = {
        'Strategy': label,
        'Final_Return': out['Cum_Strategy'].iloc[-1],
        'Sharpe': out_sharpe,
        'MDD': out_drawdown.min(),
        'Avg_Exposure': out['Executed_Total_Exposure'].mean(),
        'Turnover': out['Trade'].sum() / max(len(out), 1) * 252
    }
    return out, metrics

def backtest_strategy_variant(signal, label, use_vol_scaling=True, apply_cost=True):
    out = pd.DataFrame(index=results_df.index)
    out['Market_Return'] = results_df['Market_Return']
    out['Target_Signal'] = signal.loc[out.index].fillna(0)
    if use_vol_scaling:
        out['Target_Position_Size'] = results_df['Target_Position_Size']
    else:
        out['Target_Position_Size'] = 1.0
    out['Target_Total_Exposure'] = out['Target_Signal'] * out['Target_Position_Size']
    out['Executed_Signal'] = signal.shift(1).loc[out.index].fillna(0)
    out['Executed_Position_Size'] = out['Target_Position_Size'].shift(1).fillna(0)
    out['Executed_Total_Exposure'] = out['Target_Total_Exposure'].shift(1).fillna(0)
    out['Trade'] = out['Target_Total_Exposure'].diff().abs().fillna(out['Target_Total_Exposure'].abs())
    cost = transaction_cost if apply_cost else 0.0
    out['Strategy_Return'] = (
        out['Executed_Total_Exposure'] * out['Market_Return']
        - (out['Trade'] * cost)
    )
    out['Cum_Strategy'] = (1 + out['Strategy_Return'].fillna(0)).cumprod()
    out_peak = out['Cum_Strategy'].expanding().max()
    out_drawdown = (out['Cum_Strategy'] - out_peak) / out_peak
    out_sharpe = np.sqrt(252) * (out['Strategy_Return'].mean() / out['Strategy_Return'].std())
    metrics = {
        'Component': label,
        'Final_Return': out['Cum_Strategy'].iloc[-1],
        'Sharpe': out_sharpe,
        'MDD': out_drawdown.min(),
        'Avg_Exposure': out['Executed_Total_Exposure'].mean(),
        'Turnover': out['Trade'].sum() / max(len(out), 1) * 252
    }
    return out, metrics

logit_results_df, logit_metrics = backtest_baseline_signal(df_probs['Logit_Signal'], 'Logistic Risk Baseline')
ma200_results_df, ma200_metrics = backtest_baseline_signal(df_probs['MA200_Signal'], 'MA200 Rule Baseline')
always_in_signal = pd.Series(1.0, index=df_probs.index)
avg_exposure_target = (results_df['Executed_Signal'] * results_df['Executed_Position_Size']).mean()
xgb_signal_only_df, xgb_signal_only_metrics = backtest_strategy_variant(
    df_probs['Signal'], 'XGBoost Signal Only', use_vol_scaling=False, apply_cost=True
)
vol_only_df, vol_only_metrics = backtest_strategy_variant(
    always_in_signal, 'Vol Scaling Only', use_vol_scaling=True, apply_cost=False
)
matched_exposure_df, matched_exposure_metrics = backtest_signal_on_index(
    always_in_signal,
    results_df.index,
    use_vol_scaling=False,
    apply_cost=False,
    constant_exposure=avg_exposure_target
)
matched_exposure_metrics['Strategy'] = 'Matched Avg Exposure'
matched_exposure_metrics['Component'] = 'Matched Avg Exposure'
continuous_exposure_df, continuous_exposure_metrics = backtest_signal_on_index(
    df_probs['Exposure'],
    results_df.index,
    use_vol_scaling=True,
    apply_cost=True
)
continuous_exposure_metrics['Strategy'] = 'Continuous Overlay'
continuous_exposure_metrics['Component'] = 'Continuous Overlay'
final_variant_df, final_variant_metrics = backtest_strategy_variant(
    df_probs['Signal'], 'Final: XGBoost + Vol Scaling', use_vol_scaling=True, apply_cost=True
)
market_variant_df, market_variant_metrics = backtest_strategy_variant(
    always_in_signal, 'Buy & Hold', use_vol_scaling=False, apply_cost=False
)
attribution_comparison = pd.DataFrame([
    market_variant_metrics,
    matched_exposure_metrics,
    xgb_signal_only_metrics,
    vol_only_metrics,
    continuous_exposure_metrics,
    final_variant_metrics
])
strategy_comparison = pd.DataFrame([
    {
        'Strategy': 'XGBoost Risk Model',
        'Final_Return': results_df['Cum_Strategy'].iloc[-1],
        'Sharpe': sharpe,
        'MDD': mdd,
        'Avg_Exposure': results_df['Executed_Total_Exposure'].mean(),
        'Turnover': results_df['Trade'].sum() / max(len(results_df), 1) * 252
    },
    continuous_exposure_metrics,
    matched_exposure_metrics,
    logit_metrics,
    ma200_metrics
])

valid_best_params = [
    params for params in best_params_list
    if not any(pd.isna(list(params.values())))
]
selected_model_params = valid_best_params[-1] if valid_best_params else {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 100
}

print(f"최종 전략 누적 수익률: {results_df['Cum_Strategy'].iloc[-1]:.4f}")
print(f"시장(NASDAQ) 누적 수익률: {results_df['Cum_Market'].iloc[-1]:.4f}")
print(f"샤프 지수 (Sharpe Ratio): {sharpe:.2f}")
print(f"최대 낙폭 (MDD): {mdd:.2%}")

# 시각화
print("\n--- Baseline Comparison ---")
print(strategy_comparison.to_string(index=False, formatters={
    'Final_Return': '{:.4f}'.format,
    'Sharpe': '{:.4f}'.format,
    'MDD': '{:.2%}'.format,
    'Avg_Exposure': '{:.2%}'.format,
    'Turnover': '{:.2f}'.format
}))
print("\n--- Return Attribution ---")
print(attribution_comparison.to_string(index=False, formatters={
    'Final_Return': '{:.4f}'.format,
    'Sharpe': '{:.4f}'.format,
    'MDD': '{:.2%}'.format,
    'Avg_Exposure': '{:.2%}'.format,
    'Turnover': '{:.2f}'.format
}))

target_sensitivity_rows = []
for horizon, threshold in [(10, -0.03), (10, -0.05), (20, -0.05), (20, -0.07), (40, -0.05), (40, -0.07)]:
    target_series = (compute_true_forward_drawdown(df['NASDAQ'], horizon) <= threshold).astype(float)
    sensitivity_df = df_model.drop(columns=['Target']).copy()
    sensitivity_df['Target'] = target_series.reindex(sensitivity_df.index)
    sensitivity_df = sensitivity_df.replace([np.inf, -np.inf], np.nan).dropna()
    X_sensitivity = sensitivity_df.drop(columns=['Target', 'Forward_Drawdown'])
    y_sensitivity = sensitivity_df['Target'].astype(int)
    sensitivity_probs = run_cv_pipeline(X_sensitivity, y_sensitivity, selected_model_params)
    valid_mask = sensitivity_probs['Risk_Prob'].notna()
    if valid_mask.sum() == 0:
        continue
    sensitivity_backtest, sensitivity_metrics = backtest_signal_on_index(
        sensitivity_probs['Exposure'],
        sensitivity_probs.index[np.argmax(valid_mask.to_numpy()) + 1:],
        use_vol_scaling=True,
        apply_cost=True
    )
    target_sensitivity_rows.append({
        'Horizon': horizon,
        'Threshold': threshold,
        'Base_Rate': y_sensitivity.mean(),
        'PR_AUC': average_precision_score(y_sensitivity.loc[valid_mask], sensitivity_probs.loc[valid_mask, 'Risk_Prob']),
        'Final_Return': sensitivity_metrics['Final_Return'],
        'Sharpe': sensitivity_metrics['Sharpe'],
        'MDD': sensitivity_metrics['MDD'],
        'Avg_Exposure': sensitivity_metrics['Avg_Exposure'],
        'Turnover': sensitivity_metrics['Turnover']
    })

target_sensitivity = pd.DataFrame(target_sensitivity_rows)
target_sensitivity.to_csv(OUTPUT_DIR / 'target_sensitivity_v4.csv', index=False)

signal_sensitivity_rows = []
for ema_span, lower, upper in [(3, 0.40, 0.60), (3, 0.45, 0.55), (5, 0.45, 0.55), (10, 0.45, 0.55), (10, 0.40, 0.60), (20, 0.45, 0.55)]:
    sensitivity_probs = run_cv_pipeline(X, y, selected_model_params, ema_span=ema_span, lower=lower, upper=upper)
    valid_mask = sensitivity_probs['Risk_Prob'].notna()
    if valid_mask.sum() == 0:
        continue
    sensitivity_backtest, sensitivity_metrics = backtest_signal_on_index(
        sensitivity_probs['Exposure'],
        sensitivity_probs.index[np.argmax(valid_mask.to_numpy()) + 1:],
        use_vol_scaling=True,
        apply_cost=True
    )
    signal_sensitivity_rows.append({
        'EMA_Span': ema_span,
        'Lower_Threshold': lower,
        'Upper_Threshold': upper,
        'PR_AUC': average_precision_score(y.loc[valid_mask], sensitivity_probs.loc[valid_mask, 'Risk_Prob']),
        'Final_Return': sensitivity_metrics['Final_Return'],
        'Sharpe': sensitivity_metrics['Sharpe'],
        'MDD': sensitivity_metrics['MDD'],
        'Avg_Exposure': sensitivity_metrics['Avg_Exposure'],
        'Turnover': sensitivity_metrics['Turnover']
    })

signal_sensitivity = pd.DataFrame(signal_sensitivity_rows)
signal_sensitivity.to_csv(OUTPUT_DIR / 'signal_sensitivity_v4.csv', index=False)

diagnostic_rows = []
base_exposure = df_probs['Exposure']
base_signal = df_probs['Signal']
for ema_span, lower, upper in [(3, 0.40, 0.60), (3, 0.45, 0.55), (5, 0.45, 0.55), (10, 0.45, 0.55), (10, 0.40, 0.60), (20, 0.45, 0.55)]:
    sensitivity_probs = run_cv_pipeline(X, y, selected_model_params, ema_span=ema_span, lower=lower, upper=upper)
    aligned_exposure = sensitivity_probs['Exposure'].reindex(base_exposure.index)
    aligned_signal = sensitivity_probs['Signal'].reindex(base_signal.index)
    diagnostic_rows.append({
        'Scenario': f'ema{ema_span}_l{lower:.2f}_u{upper:.2f}',
        'Exposure_Diff_Days': int((aligned_exposure.round(6) != base_exposure.round(6)).sum()),
        'Signal_Diff_Days': int((aligned_signal != base_signal).sum()),
        'Near_40_60_Count': int(((sensitivity_probs['Risk_Prob_EMA'] >= 0.40) & (sensitivity_probs['Risk_Prob_EMA'] <= 0.60)).sum()),
        'Near_45_55_Count': int(((sensitivity_probs['Risk_Prob_EMA'] >= 0.45) & (sensitivity_probs['Risk_Prob_EMA'] <= 0.55)).sum())
    })

signal_diagnostics = pd.DataFrame(diagnostic_rows)
signal_diagnostics.to_csv(OUTPUT_DIR / 'signal_diagnostics_v4.csv', index=False)

feature_groups = {
    'Full Model': list(X.columns),
    'Liquidity Only': [col for col in X.columns if 'Reserves' in col or 'Net_Liq' in col or 'RRP' in col],
    'Macro Stress Only': [col for col in X.columns if any(key in col for key in ['VIX', 'MOVE', 'HY_Spread', 'Financial_Stress', 'Fed_Funds_Rate', '10Y_Treasury'])],
    'Trend Only': [col for col in X.columns if col in ['NASDAQ_Ret', 'Trend']]
}
feature_ablation_rows = []
for group_name, columns in feature_groups.items():
    if len(columns) == 0:
        continue
    ablation_probs = run_cv_pipeline(X[columns], y, selected_model_params)
    valid_mask = ablation_probs['Risk_Prob'].notna()
    if valid_mask.sum() == 0:
        continue
    ablation_backtest, ablation_metrics = backtest_signal_on_index(
        ablation_probs['Exposure'],
        ablation_probs.index[np.argmax(valid_mask.to_numpy()) + 1:],
        use_vol_scaling=True,
        apply_cost=True
    )
    feature_ablation_rows.append({
        'Feature_Group': group_name,
        'Num_Features': len(columns),
        'PR_AUC': average_precision_score(y.loc[valid_mask], ablation_probs.loc[valid_mask, 'Risk_Prob']),
        'Final_Return': ablation_metrics['Final_Return'],
        'Sharpe': ablation_metrics['Sharpe'],
        'MDD': ablation_metrics['MDD'],
        'Avg_Exposure': ablation_metrics['Avg_Exposure'],
        'Turnover': ablation_metrics['Turnover']
    })

feature_ablation = pd.DataFrame(feature_ablation_rows)
feature_ablation.to_csv(OUTPUT_DIR / 'feature_ablation_v4.csv', index=False)

# Transaction cost sensitivity
cost_scenarios = [0.0, 0.0005, 0.0010, 0.0015, 0.0030, 0.0050]
cost_sensitivity_rows = []
for cost in cost_scenarios:
    scenario_returns = results_df['Raw_Strategy_Return'] - (results_df['Trade'] * cost)
    scenario_metrics = calculate_return_metrics(
        scenario_returns,
        exposure=results_df['Executed_Signal'] * results_df['Executed_Position_Size'],
        trades=results_df['Trade']
    )
    scenario_metrics['Cost_Bps'] = cost * 10000
    cost_sensitivity_rows.append(scenario_metrics)

cost_sensitivity = pd.DataFrame(cost_sensitivity_rows)[
    ['Cost_Bps', 'Final_Return', 'Sharpe', 'MDD', 'Avg_Exposure', 'Turnover']
]
cost_sensitivity.to_csv(OUTPUT_DIR / 'cost_sensitivity_v4.csv', index=False)

print("\n--- Transaction Cost Sensitivity ---")
print(cost_sensitivity.to_string(index=False, formatters={
    'Cost_Bps': '{:.0f}'.format,
    'Final_Return': '{:.4f}'.format,
    'Sharpe': '{:.4f}'.format,
    'MDD': '{:.2%}'.format,
    'Avg_Exposure': '{:.2%}'.format,
    'Turnover': '{:.2f}'.format
}))

# Stress-period analysis using the already-computed out-of-sample return streams.
def append_stress_row(rows, period_name, period_type, period_df):
    if period_df.empty:
        return

    strategy_metrics = calculate_return_metrics(
        period_df['Strategy_Return'],
        exposure=period_df['Executed_Total_Exposure'],
        trades=period_df['Trade']
    )
    market_metrics = calculate_return_metrics(period_df['Market_Return'])
    rows.append({
        'Type': period_type,
        'Period': period_name,
        'Start': period_df.index.min().strftime('%Y-%m-%d'),
        'End': period_df.index.max().strftime('%Y-%m-%d'),
        'Strategy_Return': strategy_metrics['Final_Return'],
        'Market_Return': market_metrics['Final_Return'],
        'Strategy_Sharpe': strategy_metrics['Sharpe'],
        'Market_Sharpe': market_metrics['Sharpe'],
        'Strategy_MDD': strategy_metrics['MDD'],
        'Market_MDD': market_metrics['MDD'],
        'Avg_Exposure': strategy_metrics['Avg_Exposure'],
        'Turnover': strategy_metrics['Turnover']
    })

stress_rows = []

stress_periods = {
    'OOS AI / High-Rate Cycle': (results_df.index.min(), min(pd.Timestamp('2025-12-31'), results_df.index.max())),
    'Recent OOS Window': (max(results_df.index.min(), results_df.index.max() - pd.Timedelta(days=365)), results_df.index.max())
}

for period_name, (start, end) in stress_periods.items():
    period_mask = (results_df.index >= pd.Timestamp(start)) & (results_df.index <= pd.Timestamp(end))
    append_stress_row(stress_rows, period_name, 'Predefined Stress', results_df.loc[period_mask])

window = min(63, len(results_df))
if window > 5:
    market_window_return = (1 + results_df['Market_Return'].fillna(0)).rolling(window).apply(np.prod, raw=True) - 1
    strategy_window_return = (1 + results_df['Strategy_Return'].fillna(0)).rolling(window).apply(np.prod, raw=True) - 1
    worst_market_end = market_window_return.idxmin()
    worst_strategy_end = strategy_window_return.idxmin()
    worst_market_start = results_df.index[max(results_df.index.get_loc(worst_market_end) - window + 1, 0)]
    worst_strategy_start = results_df.index[max(results_df.index.get_loc(worst_strategy_end) - window + 1, 0)]
    append_stress_row(
        stress_rows,
        f'Worst Market {window}D Window',
        'Worst Window',
        results_df.loc[worst_market_start:worst_market_end]
    )
    append_stress_row(
        stress_rows,
        f'Worst Strategy {window}D Window',
        'Worst Window',
        results_df.loc[worst_strategy_start:worst_strategy_end]
    )

stress_analysis = pd.DataFrame(stress_rows)
stress_analysis.to_csv(OUTPUT_DIR / 'stress_analysis_v4.csv', index=False)

print("\n--- Stress Period Analysis ---")
print(stress_analysis.to_string(index=False, formatters={
    'Strategy_Return': '{:.4f}'.format,
    'Market_Return': '{:.4f}'.format,
    'Strategy_Sharpe': '{:.4f}'.format,
    'Market_Sharpe': '{:.4f}'.format,
    'Strategy_MDD': '{:.2%}'.format,
    'Market_MDD': '{:.2%}'.format,
    'Avg_Exposure': '{:.2%}'.format,
    'Turnover': '{:.2f}'.format
}))

plt.figure(figsize=(12, 6))
plt.plot(results_df['Cum_Market'], label='NASDAQ (Buy & Hold)', color='gray', alpha=0.5)
plt.plot(results_df['Cum_Strategy'], label='Downside-Risk Overlay (v4)', color='red', linewidth=2)
plt.plot(logit_results_df['Cum_Strategy'], label='Logistic Risk Baseline', color='blue', linewidth=1.5, alpha=0.8)
plt.plot(ma200_results_df['Cum_Strategy'], label='MA200 Rule Baseline', color='green', linewidth=1.5, alpha=0.8)
plt.plot(xgb_signal_only_df['Cum_Strategy'], label='XGBoost Signal Only', color='orange', linewidth=1.2, alpha=0.7)
plt.plot(vol_only_df['Cum_Strategy'], label='Vol Scaling Only', color='purple', linewidth=1.2, alpha=0.7)
plt.title("Backtest v4: Downside-Risk Target & Hold Signals")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / 'final_backtesting_v4.png')

print("\n" + "="*50)
print("FINAL SUMMARY (Strategy v4)")
print("="*50)
print(f"PR-AUC: {xgb_pr_auc:.4f}")
print(f"Brier Score: {xgb_brier:.4f}")
print(f"Log Loss: {xgb_log_loss:.4f}")
print(f"Accuracy (secondary): {accuracy_score(test_actual, test_pred):.4f}")
print(f"Risk-Event Base Rate: {y.mean():.4f}")
print(f"Final Strategy Return: {results_df['Cum_Strategy'].iloc[-1]:.4f}")
print(f"Market (NASDAQ) Return: {results_df['Cum_Market'].iloc[-1]:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {mdd:.2%}")
print("-" * 50)
print("Top Liquidity Beta Stocks:")
print(df_beta.head(5))
print("="*50)

