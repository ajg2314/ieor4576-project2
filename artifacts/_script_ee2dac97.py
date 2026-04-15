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
conn = sqlite3.connect('artifacts/eda_b37ac91d.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd

# Load all 10-K revenue data
revenue_df = df[(df['metric'] == 'revenue') & (df['form'] == '10-K')].copy()

# Find the most recent fiscal year for each company
most_recent_years = revenue_df.groupby('ticker')['fiscal_year'].max().reset_index()
most_recent_years.rename(columns={'fiscal_year': 'max_fiscal_year'}, inplace=True)

# Merge back to get the revenue for the most recent year
recent_revenue_df = pd.merge(revenue_df, most_recent_years, on='ticker', how='inner')
recent_revenue_df = recent_revenue_df[recent_revenue_df['fiscal_year'] == recent_revenue_df['max_fiscal_year']]

# Select relevant columns and sort
ranked_companies = recent_revenue_df[['ticker', 'company', 'fiscal_year', 'value_billions']].sort_values(by='value_billions', ascending=False)

print("Most Recent Fiscal Year Revenue Ranking:")
print(ranked_companies.to_markdown(index=False))
