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
conn = sqlite3.connect('artifacts/eda_c6f74ef1.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd

# Fetch all 10-K revenue data for the specified companies
df_filtered = df[(df['metric'] == 'revenue') & (df['form'] == '10-K') & (df['ticker'].isin(['NVDA', 'AMD', 'INTC']))]

# Convert fiscal_year to numeric and sort
df_filtered['fiscal_year'] = pd.to_numeric(df_filtered['fiscal_year'])
df_filtered = df_filtered.sort_values(by=['ticker', 'fiscal_year'])

# --- 1. Most recent year ranking ---
print("--- Most Recent Fiscal Year Revenue ---")
most_recent_revenue = df_filtered.groupby('ticker').apply(lambda x: x.loc[x['fiscal_year'].idxmax()])
print(most_recent_revenue[['ticker', 'fiscal_year', 'value_billions']].sort_values(by='value_billions', ascending=False))
print("\n")

# --- 2. YoY Revenue Growth % ---
print("--- Year-over-Year Revenue Growth % ---")
yoy_growth_data = []
for ticker in df_filtered['ticker'].unique():
    company_df = df_filtered[df_filtered['ticker'] == ticker].copy()
    company_df['YoY Growth %'] = company_df['value_billions'].pct_change() * 100
    # Get only the most recent YoY growth that is not NaN
    last_yoy_growth = company_df.dropna(subset=['YoY Growth %']).tail(1)
    if not last_yoy_growth.empty:
        yoy_growth_data.append({
            'ticker': ticker,
            'fiscal_year': last_yoy_growth['fiscal_year'].iloc[0],
            'YoY Growth %': last_yoy_growth['YoY Growth %'].iloc[0]
        })
yoy_df = pd.DataFrame(yoy_growth_data).sort_values(by='YoY Growth %', ascending=False)
print(yoy_df)
print("\n")

# --- 3. 3-year and 5-year revenue CAGR ---
print("--- 3-Year and 5-Year Revenue CAGR % ---")
cagr_data = []
for ticker in df_filtered['ticker'].unique():
    company_df = df_filtered[df_filtered['ticker'] == ticker].copy()
    company_df = company_df.sort_values(by='fiscal_year')

    # Ensure there's enough data for CAGR calculation
    if len(company_df) >= 3:
        # 3-year CAGR
        end_val_3yr = company_df.iloc[-1]['value_billions']
        start_val_3yr = company_df.iloc[-3]['value_billions']
        if start_val_3yr > 0: # Avoid division by zero
            cagr_3yr = ((end_val_3yr / start_val_3yr)**(1/2) - 1) * 100
        else:
            cagr_3yr = float('nan') # Or handle as appropriate if start_val is zero/negative

    else:
        cagr_3yr = float('nan')

    if len(company_df) >= 5:
        # 5-year CAGR
        end_val_5yr = company_df.iloc[-1]['value_billions']
        start_val_5yr = company_df.iloc[-5]['value_billions']
        if start_val_5yr > 0: # Avoid division by zero
            cagr_5yr = ((end_val_5yr / start_val_5yr)**(1/4) - 1) * 100
        else:
            cagr_5yr = float('nan')
    else:
        cagr_5yr = float('nan')

    cagr_data.append({
        'ticker': ticker,
        '3-Year CAGR %': cagr_3yr,
        '5-Year CAGR %': cagr_5yr
    })
cagr_df = pd.DataFrame(cagr_data)
print(cagr_df)
print("\n")

# --- 4. Linear regression of revenue vs year ---
print("--- Linear Regression of Revenue vs. Year (Slope) ---")
from scipy.stats import linregress

regression_data = []
for ticker in df_filtered['ticker'].unique():
    company_df = df_filtered[df_filtered['ticker'] == ticker].copy()
    if len(company_df) > 1: # Need at least 2 points for regression
        slope, intercept, r_value, p_value, std_err = linregress(company_df['fiscal_year'], company_df['value_billions'])
        regression_data.append({'ticker': ticker, 'Slope (Revenue/Year)': slope})
    else:
        regression_data.append({'ticker': ticker, 'Slope (Revenue/Year)': float('nan')})
regression_df = pd.DataFrame(regression_data).sort_values(by='Slope (Revenue/Year)', ascending=False)
print(regression_df)
print("\n")
