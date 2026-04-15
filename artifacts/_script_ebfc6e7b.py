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
conn = sqlite3.connect('artifacts/eda_8d2d2581.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Filter for annual revenue data
revenue_df = df[(df['metric'] == 'revenue') & (df['form'] == '10-K')].copy()
revenue_df['fiscal_year'] = pd.to_numeric(revenue_df['fiscal_year'])

# Pivot table for easier calculations
revenue_pivot = revenue_df.pivot_table(index='ticker', columns='fiscal_year', values='value_billions')

print("--- YoY Revenue Growth (%) ---")
yoy_growth = revenue_pivot.pct_change(axis=1) * 100
print(yoy_growth.tail(5)) # Displaying tail to show recent years for brevity, can adjust

print("\n--- 3-Year Revenue CAGR (%) ---")
cagr_3yr = {}
for ticker in revenue_pivot.index:
    years_data = revenue_pivot.loc[ticker].dropna().sort_index()
    if len(years_data) >= 3:
        end_val = years_data.iloc[-1]
        start_val = years_data.iloc[-3]
        if start_val > 0:
            cagr = ((end_val / start_val)**(1/2) - 1) * 100 # 2 periods for 3 years
            cagr_3yr[ticker] = round(cagr, 2)
cagr_3yr_df = pd.DataFrame.from_dict(cagr_3yr, orient='index', columns=['CAGR_3Y'])
print(cagr_3yr_df.sort_values(by='CAGR_3Y', ascending=False))

print("\n--- 5-Year Revenue CAGR (%) ---")
cagr_5yr = {}
for ticker in revenue_pivot.index:
    years_data = revenue_pivot.loc[ticker].dropna().sort_index()
    if len(years_data) >= 5:
        end_val = years_data.iloc[-1]
        start_val = years_data.iloc[-5]
        if start_val > 0:
            cagr = ((end_val / start_val)**(1/4) - 1) * 100 # 4 periods for 5 years
            cagr_5yr[ticker] = round(cagr, 2)
cagr_5yr_df = pd.DataFrame.from_dict(cagr_5yr, orient='index', columns=['CAGR_5Y'])
print(cagr_5yr_df.sort_values(by='CAGR_5Y', ascending=False))

print("\n--- Revenue Linear Regression (Slope) ---")
regression_slopes = {}
for ticker in revenue_pivot.index:
    years_data = revenue_pivot.loc[ticker].dropna()
    if len(years_data) > 1:
        X = years_data.index.values.reshape(-1, 1)
        y = years_data.values
        model = LinearRegression()
        model.fit(X, y)
        regression_slopes[ticker] = round(model.coef_[0], 2) # Slope represents annual growth in billions
regression_slopes_df = pd.DataFrame.from_dict(regression_slopes, orient='index', columns=['Regression_Slope_Billions'])
print(regression_slopes_df.sort_values(by='Regression_Slope_Billions', ascending=False))
