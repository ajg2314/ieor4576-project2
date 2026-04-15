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
conn = sqlite3.connect('artifacts/eda_5a270b1c.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd
import numpy as np

# Load revenue data
revenue_df = df[(df['metric'] == 'revenue') & (df['form'] == '10-K')].copy()
revenue_df = revenue_df.pivot_table(index='fiscal_year', columns='ticker', values='value_billions')

# Sort by fiscal_year to ensure correct calculation
revenue_df.sort_index(inplace=True)

# Calculate YoY Growth
yoy_growth = revenue_df.pct_change() * 100

print("### Year-over-Year Revenue Growth (%)")
print(yoy_growth.tail(5).round(2))

# Calculate CAGR
def calculate_cagr(series, years):
    if len(series) < years + 1:
        return np.nan
    end_val = series.iloc[-1]
    start_val = series.iloc[-(years + 1)]
    if start_val <= 0: # Avoid division by zero or negative values for CAGR calculation
        return np.nan
    return ((end_val / start_val)**(1/years) - 1) * 100

cagr_data = {}
for ticker in revenue_df.columns:
    cagr_data[ticker] = {
        '3-Year CAGR (%)': calculate_cagr(revenue_df[ticker], 3),
        '5-Year CAGR (%)': calculate_cagr(revenue_df[ticker], 5)
    }

cagr_df = pd.DataFrame(cagr_data).T
print("\n### Compound Annual Growth Rate (CAGR) %")
print(cagr_df.round(2))

# Linear Regression for Revenue Momentum
from scipy.stats import linregress

regression_results = {}
for ticker in revenue_df.columns:
    company_data = revenue_df[ticker].dropna()
    if len(company_data) > 1: # Need at least 2 points for regression
        x = company_data.index.astype(int)
        y = company_data.values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_results[ticker] = {
            'Slope (Billion/Year)': slope,
            'R-squared': r_value**2
        }
    else:
        regression_results[ticker] = {
            'Slope (Billion/Year)': np.nan,
            'R-squared': np.nan
        }

regression_df = pd.DataFrame(regression_results).T
print("\n### Revenue Linear Regression Slope (Momentum)")
print(regression_df.round(2))
