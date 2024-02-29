import numpy as np
import pandas as pd

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Train-test split function
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(test_size * len(X))
    test_indices, train_indices = indices[:test_size], indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Logistic regression training function
def train(X_train, y_train, lr, epochs):
    # Initialize weights
    np.random.seed(42)
    w = np.random.rand(X_train.shape[1])
    b = 0
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        z = np.dot(X_train, w) + b
        a = sigmoid(z)
        
        # Compute loss
        loss = -np.mean(y_train * np.log(a) + (1 - y_train) * np.log(1 - a))
        
        # Backpropagation
        dw = np.dot(X_train.T, (a - y_train)) / len(X_train)
        db = np.mean(a - y_train)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
    
    return w, b

# Prediction function
def predict(X_test, w, b):
    z = np.dot(X_test, w) + b
    a = sigmoid(z)
    return (a > 0.5).astype(int)

# Accuracy calculation function
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# Load data
data = pd.read_csv(r"data_train.csv")
# Check for missing values
missing_values = data.isnull().any(axis=1)

# Get the instances with missing values
removed_instances = data[missing_values]

if removed_instances.empty:
    print("No missing values")
else:
    print("Instances with missing values removed:")
    print(removed_instances)

# Convert categorical target variable to numerical
data['species'] = (data['species'] == 'Setosa').astype(int)

# Split data into features and target variable
X = data.drop('species', axis=1).values
y = data['species'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
w, b = train(X_train, y_train, lr=0.01, epochs=92)

# Make predictions
predictions = predict(X_test, w, b)

# Calculate accuracy
accuracy = calculate_accuracy(y_test, predictions)
print("Accuracy:", accuracy)



