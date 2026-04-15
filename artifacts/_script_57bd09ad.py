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
conn = sqlite3.connect('artifacts/eda_378b6d5c.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd

# Filter for relevant tickers and annual revenue data
revenue_df = df[(df['ticker'].isin(['LLY', 'PFE', 'MRK', 'ABBV'])) & (df['metric'] == 'revenue') & (df['form'] == '10-K')].copy()

# Ensure fiscal_year is numeric for sorting
revenue_df['fiscal_year'] = pd.to_numeric(revenue_df['fiscal_year'])

# Sort by ticker and fiscal_year
revenue_df = revenue_df.sort_values(by=['ticker', 'fiscal_year'])

# Calculate Year-over-Year (YoY) growth
revenue_df['YoY_Growth'] = revenue_df.groupby('ticker')['value_billions'].pct_change() * 100

print("Year-over-Year Revenue Growth (%):")
print(revenue_df[revenue_df['YoY_Growth'].notna()][['ticker', 'fiscal_year', 'value_billions', 'YoY_Growth']].round(2).to_markdown(index=False))

