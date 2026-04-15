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
conn = sqlite3.connect('artifacts/eda_a2492f3f.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd

# Load financial data
revenue_data = df[(df['metric'] == 'revenue') & (df['form'] == '10-K')].copy()
revenue_data['fiscal_year'] = pd.to_numeric(revenue_data['fiscal_year'])

# Pivot data to have years as columns for easier calculation
revenue_pivot = revenue_data.pivot_table(index='ticker', columns='fiscal_year', values='value_billions')

# Calculate YoY growth rates
yoy_growth = revenue_pivot.pct_change(axis=1) * 100

# Get the most recent YoY growth
most_recent_year = revenue_pivot.columns.max()
second_most_recent_year = revenue_pivot.columns[-2]

# Display YoY growth for the last few years and the most recent year
print(f"\n--- Annual Revenue YoY Growth (%) ---\n")
print(yoy_growth.tail(5).round(2))

print(f"\n--- Most Recent Annual Revenue YoY Growth (ending {most_recent_year}) (%) ---\n")
print(yoy_growth[[most_recent_year]].sort_values(by=most_recent_year, ascending=False).round(2))
