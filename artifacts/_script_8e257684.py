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
conn = sqlite3.connect('artifacts/eda_7df91d8c.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd
import numpy as np
from scipy.stats import linregress

# Filter for annual data (10-K or yfinance-annual) and 'revenue' metric
annual_revenue_df = df[(df['metric'] == 'revenue') & (df['form'].isin(['10-K', 'yfinance-annual']))].copy()
annual_revenue_df['fiscal_year'] = pd.to_numeric(annual_revenue_df['fiscal_year'])
annual_revenue_df = annual_revenue_df.sort_values(by=['ticker', 'fiscal_year'])

print("--- YoY Revenue Growth (%) ---")
yoy_growth = annual_revenue_df.groupby('ticker')['value_billions'].pct_change() * 100
annual_revenue_df['YoY_Growth'] = yoy_growth
# Filter out the first year for each company as YoY growth will be NaN
print(annual_revenue_df[~annual_revenue_df['YoY_Growth'].isna()][['company', 'fiscal_year', 'value_billions', 'YoY_Growth']].to_string())

print("\n--- 3-Year and 5-Year Revenue CAGR (%) ---")
cagr_data = []
for ticker, group in annual_revenue_df.groupby('ticker'):
    group = group.sort_values(by='fiscal_year')

    # 3-year CAGR
    if len(group) >= 3:
        end_val_3yr = group['value_billions'].iloc[-1]
        start_val_3yr = group['value_billions'].iloc[-3]
        if start_val_3yr > 0:
            cagr_3yr = ((end_val_3yr / start_val_3yr)**(1/2) - 1) * 100
            cagr_data.append({'company': group['company'].iloc[0], 'ticker': ticker, 'CAGR_Type': '3-Year Revenue CAGR', 'CAGR_Value': cagr_3yr})
        else:
            cagr_data.append({'company': group['company'].iloc[0], 'ticker': ticker, 'CAGR_Type': '3-Year Revenue CAGR', 'CAGR_Value': float('nan')})
    else:
        cagr_data.append({'company': group['company'].iloc[0], 'ticker': ticker, 'CAGR_Type': '3-Year Revenue CAGR', 'CAGR_Value': float('nan')})


    # 5-year CAGR
    if len(group) >= 5:
        end_val_5yr = group['value_billions'].iloc[-1]
        start_val_5yr = group['value_billions'].iloc[-5]
        if start_val_5yr > 0:
            cagr_5yr = ((end_val_5yr / start_val_5yr)**(1/4) - 1) * 100
            cagr_data.append({'company': group['company'].iloc[0], 'ticker': ticker, 'CAGR_Type': '5-Year Revenue CAGR', 'CAGR_Value': cagr_5yr})
        else:
            cagr_data.append({'company': group['company'].iloc[0], 'ticker': ticker, 'CAGR_Type': '5-Year Revenue CAGR', 'CAGR_Value': float('nan')})
    else:
        cagr_data.append({'company': group['company'].iloc[0], 'ticker': ticker, 'CAGR_Type': '5-Year Revenue CAGR', 'CAGR_Value': float('nan')})

cagr_df = pd.DataFrame(cagr_data)
print(cagr_df.to_string())

print("\n--- Linear Regression of Revenue vs Year (Slope) ---")
regression_results = []
for ticker, group in annual_revenue_df.groupby('ticker'):
    if len(group) >= 2: # Need at least two points for regression
        # Use fiscal_year as x and value_billions as y
        x = group['fiscal_year']
        y = group['value_billions']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_results.append({'company': group['company'].iloc[0], 'ticker': ticker, 'Regression_Slope': slope, 'R_Squared': r_value**2})
    else:
        regression_results.append({'company': group['company'].iloc[0], 'ticker': ticker, 'Regression_Slope': float('nan'), 'R_Squared': float('nan')})

regression_df = pd.DataFrame(regression_results)
print(regression_df.to_string())
