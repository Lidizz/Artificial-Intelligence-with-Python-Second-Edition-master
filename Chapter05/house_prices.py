import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# Load housing data (using California housing dataset as replacement for deprecated Boston dataset)
data = datasets.fetch_california_housing() 

# Shuffle the data
X, y = shuffle(data.data, data.target, random_state=7)

# Use a subset of data for faster training (California dataset is large: 20,640 samples)
# Using 5000 samples is sufficient for demonstration
max_samples = 5000
X = X[:max_samples]
y = y[:max_samples]

# Split the data into training and testing datasets 
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Feature scaling is critical for SVR performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Support Vector Regression model
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

# Train Support Vector Regressor
sv_regressor.fit(X_train, y_train)

# Evaluate performance of Support Vector Regressor
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred) 
print("\n#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Test the regressor on test datapoint
# California housing features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
test_data = [8.3252, 41.0, 6.98, 1.02, 322.0, 2.56, 37.88, -122.23]
test_data_scaled = scaler.transform([test_data])
print("\nPredicted price (in $100,000s):", sv_regressor.predict(test_data_scaled)[0])

