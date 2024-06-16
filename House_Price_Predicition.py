import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras

# Example dataset
data = [
    {'size': 1500, 'bedrooms': 3, 'bathrooms': 2, 'price': 300000},
    {'size': 2000, 'bedrooms': 4, 'bathrooms': 3, 'price': 450000},
    {'size': 1200, 'bedrooms': 2, 'bathrooms': 1, 'price': 200000},
    {'size': 1800, 'bedrooms': 3, 'bathrooms': 2, 'price': 350000},
    {'size': 2200, 'bedrooms': 4, 'bathrooms': 3, 'price': 500000},
    # Add more data as needed
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Split features and target
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)

# Predict the house prices on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Example: Predicting prices for new data
new_data = np.array([[3000, 3, 2]])  # Example new data
new_data = scaler.transform(new_data)  # Don't forget to scale new data
predicted_price = model.predict(new_data)
print('Predicted Price:', predicted_price)

