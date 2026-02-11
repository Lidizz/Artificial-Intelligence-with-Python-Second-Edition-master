import datetime
import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster

import yfinance as yf


# Input file containing company symbols 
input_file = 'company_symbol_mapping.json'

# Load the company symbol map
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Load the historical stock quotes 
start_date = "2025-01-01"
end_date = "2025-02-28"

# Use yf.download for batch downloading which is faster and more reliable
# We need to handle cases where data might be missing
symbols_list = list(symbols)
data = yf.download(symbols_list, start=start_date, end=end_date)

# Extract opening and closing quotes
# yfinance returns a MultiIndex DataFrame, we need to extract Open and Close properly
# The columns are sorted by symbol, so we need to align our names array
downloaded_symbols = data.columns.get_level_values(1).unique() if isinstance(data.columns, pd.MultiIndex) else data.columns
# Note: yfinance 1.1.0 might return data slightly differently depending on structure.
# But generally data['Open'] has columns as symbols.
opening_quotes = data['Open'].T.to_numpy()
closing_quotes = data['Close'].T.to_numpy()

# Re-align names to match the sorted symbols in data
# Create a dictionary for easy lookup
symbol_to_name = dict(zip(symbols, names))
# Get the symbols in the order they appear in the dataframe columns
# data['Open'].columns contains the symbols
current_symbols = data['Open'].columns
names = np.array([symbol_to_name.get(sym, sym) for sym in current_symbols])
symbols = np.array(current_symbols)

# Filter out companies with missing data (NaN)
valid_indices = ~np.isnan(opening_quotes).any(axis=1) & ~np.isnan(closing_quotes).any(axis=1)
opening_quotes = opening_quotes[valid_indices]
closing_quotes = closing_quotes[valid_indices]
symbols = symbols[valid_indices]
names = names[valid_indices]

# Compute differences between opening and closing quotes 
quotes_diff = closing_quotes - opening_quotes

# Normalize the data 
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Create a graph model 
edge_model = covariance.GraphicalLassoCV()

# Train the model
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Build clustering model using Affinity Propagation model
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Print the results of clustering
print('\nClustering of stocks based on difference in opening and closing quotes:\n')
for i in range(num_labels + 1):
    print("Cluster", i+1, "==>", ', '.join(names[labels == i]))
