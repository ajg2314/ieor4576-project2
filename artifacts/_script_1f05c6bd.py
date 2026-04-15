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
conn = sqlite3.connect('artifacts/eda_67df0681.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Filter for annual 10-K revenue data
revenue_df = df[(df['metric'] == 'revenue') & (df['form'] == '10-K')].copy()
revenue_df['fiscal_year'] = pd.to_numeric(revenue_df['fiscal_year'])
revenue_df = revenue_df.sort_values(by=['ticker', 'fiscal_year'])

print("--- YoY Revenue Growth % ---")
yoy_growth = revenue_df.groupby('ticker')['value_billions'].pct_change() * 100
revenue_df['YoY_Growth'] = yoy_growth
latest_yoy = revenue_df.dropna(subset=['YoY_Growth']).groupby('ticker').apply(lambda x: x.loc[x['fiscal_year'].idxmax()])
print(latest_yoy[['fiscal_year', 'value_billions', 'YoY_Growth']].sort_values(by='YoY_Growth', ascending=False).to_markdown(index=False))

print("\n--- 3-Year and 5-Year Revenue CAGR % ---")
cagr_data = []
for ticker in revenue_df['ticker'].unique():
    company_df = revenue_df[revenue_df['ticker'] == ticker].sort_values(by='fiscal_year')
    max_year = company_df['fiscal_year'].max()

    # 3-year CAGR
    if max_year - 2 in company_df['fiscal_year'].values:
        end_val_3yr = company_df[company_df['fiscal_year'] == max_year]['value_billions'].iloc[0]
        start_val_3yr = company_df[company_df['fiscal_year'] == max_year - 2]['value_billions'].iloc[0]
        cagr_3yr = ((end_val_3yr / start_val_3yr)**(1/2) - 1) * 100 if start_val_3yr != 0 else np.nan
    else:
        cagr_3yr = np.nan

    # 5-year CAGR
    if max_year - 4 in company_df['fiscal_year'].values:
        end_val_5yr = company_df[company_df['fiscal_year'] == max_year]['value_billions'].iloc[0]
        start_val_5yr = company_df[company_df['fiscal_year'] == max_year - 4]['value_billions'].iloc[0]
        cagr_5yr = ((end_val_5yr / start_val_5yr)**(1/4) - 1) * 100 if start_val_5yr != 0 else np.nan
    else:
        cagr_5yr = np.nan
    cagr_data.append({'ticker': ticker, '3_Year_CAGR': cagr_3yr, '5_Year_CAGR': cagr_5yr})

cagr_df = pd.DataFrame(cagr_data)
print(cagr_df.sort_values(by='3_Year_CAGR', ascending=False).to_markdown(index=False))

print("\n--- Linear Regression of Revenue vs Year (Slope) ---")
regression_results = []
for ticker in revenue_df['ticker'].unique():
    company_df = revenue_df[revenue_df['ticker'] == ticker].dropna(subset=['value_billions'])
    if len(company_df) > 1: # Need at least 2 points for a line
        X = company_df['fiscal_year'].values.reshape(-1, 1)
        y = company_df['value_billions'].values
        model = LinearRegression()
        model.fit(X, y)
        regression_results.append({'ticker': ticker, 'Revenue_Regression_Slope': model.coef_[0]})
    else:
        regression_results.append({'ticker': ticker, 'Revenue_Regression_Slope': np.nan})

regression_df = pd.DataFrame(regression_results)
print(regression_df.sort_values(by='Revenue_Regression_Slope', ascending=False).to_markdown(index=False))
