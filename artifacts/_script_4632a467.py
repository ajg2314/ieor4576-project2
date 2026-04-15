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
conn = sqlite3.connect('artifacts/eda_c90b16dc.db')
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

import pandas as pd

revenue_df = df[(df['metric'] == 'revenue') & (df['form'] == '10-K')].pivot_table(
    index='fiscal_year', columns='ticker', values='value_billions'
)

# Sort by fiscal_year to ensure correct chronological order for YoY calculation
revenue_df = revenue_df.sort_index()

yoy_growth = revenue_df.pct_change() * 100
print("Year-over-Year Revenue Growth (%):")
print(yoy_growth.tail())
