import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import json, os, sys
ARTIFACTS_DIR = '/Users/guandyjay/Desktop/ieor4576-project2/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

import sqlite3, pandas as pd, numpy as np
from scipy import stats
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    pass
conn = sqlite3.connect('artifacts/eda_d00905a8.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd

# Filter for 10-K annual data for the specified tickers
df_annual = df[(df['form'] == '10-K') & (df['ticker'].isin(['XOM', 'CVX', 'COP']))].copy()
df_annual['fiscal_year'] = pd.to_numeric(df_annual['fiscal_year'])

# --- Revenue YoY Growth ---
revenue_df = df_annual[df_annual['metric'] == 'revenue'].pivot_table(index=['fiscal_year'], columns='ticker', values='value_billions')
revenue_yoy_growth = revenue_df.pct_change() * 100
print("--- Year-over-Year Revenue Growth (10-K, %) ---")
print(revenue_yoy_growth.tail(5))

# --- 3-Year and 5-Year Revenue CAGR ---
def calculate_cagr(series, years):
    if len(series) < years:
        return float('nan')
    return (series.iloc[-1] / series.iloc[-years])**(1/(years-1)) - 1 if series.iloc[-years] != 0 else float('inf')

cagr_data = {}
for ticker in revenue_df.columns:
    cagr_data[ticker] = {
        '3_Year_CAGR': calculate_cagr(revenue_df[ticker].dropna(), 3),
        '5_Year_CAGR': calculate_cagr(revenue_df[ticker].dropna(), 5)
    }
cagr_df = pd.DataFrame(cagr_data).T * 100
print("\n--- Revenue CAGR (%, last available period) ---")
print(cagr_df)

# --- Linear Regression of Revenue vs Year ---
from sklearn.linear_model import LinearRegression

regression_results = []
for ticker in revenue_df.columns:
    temp_df = revenue_df[[ticker]].dropna().reset_index()
    if len(temp_df) > 1:
        X = temp_df['fiscal_year'].values.reshape(-1, 1)
        y = temp_df[ticker].values
        model = LinearRegression()
        model.fit(X, y)
        regression_results.append({'ticker': ticker, 'slope': model.coef_[0], 'intercept': model.intercept_})

regression_df = pd.DataFrame(regression_results)
print("\n--- Revenue Linear Regression Slopes (Billions/Year) ---")
print(regression_df)

