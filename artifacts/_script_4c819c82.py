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
conn = sqlite3.connect('artifacts/eda_44fab2a7.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd

# Filter for 10-K annual revenue data for the specified companies
companies_of_interest = ['MSFT', 'GOOG', 'AMZN', 'ADBE', 'CRM', 'IBM', 'INTU', 'NET', 'NOW', 'ORCL', 'SAP']
revenue_df = df[(df['metric'] == 'revenue') & (df['form'] == '10-K') & (df['ticker'].isin(companies_of_interest))].copy()

# Ensure fiscal_year is numeric for sorting
revenue_df['fiscal_year'] = pd.to_numeric(revenue_df['fiscal_year'])
revenue_df = revenue_df.sort_values(by=['ticker', 'fiscal_year'])

# Calculate Year-over-Year (YoY) Growth
yoy_growth = revenue_df.groupby('ticker')['value_billions'].pct_change() * 100
yoy_growth_df = revenue_df.copy()
yoy_growth_df['YoY Growth %'] = yoy_growth

# Select relevant columns and pivot for better readability
yoy_growth_pivot = yoy_growth_df.pivot(index='fiscal_year', columns='ticker', values='YoY Growth %')

print("Year-over-Year Revenue Growth (%):")
print(yoy_growth_pivot.tail(5).round(2))
