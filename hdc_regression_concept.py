import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# HDC Parameters
DIMENSIONS = 10000      # High-dimensional space size
BINARY_MODE = False     # If True, uses binary hypervectors (0,1); otherwise, bipolar (-1,1)
RIDGE_ALPHA = 1.0       # Regularization strength for Ridge regression

# Helper Function: Generate a Random Base Hypervector
def generate_hypervector(dim, binary_mode=False):
    if binary_mode:
        return np.random.choice([0, 1], size=dim)
    else:
        return np.random.choice([-1, 1], size=dim)

# Generate a consistent set of base hypervectors for all features
def generate_base_hypervectors(num_features, dim, binary_mode=False):
    return np.array([generate_hypervector(dim, binary_mode) for _ in range(num_features)])

# Encoding Function: Convert feature vectors into HDC hypervectors using the provided base hypervectors
def encode_features(features, base_hypervectors):
    num_samples = features.shape[0]
    dim = base_hypervectors.shape[1]
    encoded_hvs = np.zeros((num_samples, dim))
    for i in range(num_samples):
        # Each sample is encoded by summing the weighted base hypervectors
        encoded = np.sum(features[i, :, None] * base_hypervectors, axis=0)
        # Normalize the hypervector (L2 normalization)
        norm = np.linalg.norm(encoded)
        if norm != 0:
            encoded_hvs[i] = encoded / norm
        else:
            encoded_hvs[i] = encoded
    return encoded_hvs

# Training Function: Train an HDC Regression Model using Ridge Regression
def train_hdc_regression(X_train_encoded, y_train, alpha=RIDGE_ALPHA):
    model = Ridge(alpha=alpha)
    model.fit(X_train_encoded, y_train)
    return model

# Prediction Function: Make Continuous Predictions using the trained regression model
def predict_hdc_regression(model, X_test_encoded):
    predictions = model.predict(X_test_encoded)
    return predictions

# Evaluation Function: Compute and Print Model Metrics
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("Evaluation Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")
    return mse, mae, r2

# Plotting Function: Visualize Actual vs. Predicted Values
def plot_results(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Predictions")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label="Ideal")
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('HDC Regression: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():

    # Generate Regression Data
    num_samples = 10000
    num_features = 10
    X = np.random.rand(num_samples, num_features)  # Features in range [0, 1]
    y = (3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, num_samples))     # Linear regression target with noise

    # Scale Data using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Data into Training and Test Sets (80% train, 20% test)
    split_index = int(num_samples * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Generate consistent base hypervectors for all features
    base_hypervectors = generate_base_hypervectors(num_features, DIMENSIONS, BINARY_MODE)

    # Encode Training and Test Features using the same base hypervectors
    X_train_encoded = encode_features(X_train, base_hypervectors)
    X_test_encoded = encode_features(X_test, base_hypervectors)

    # Train HDC Regression Model using Ridge Regression
    model = train_hdc_regression(X_train_encoded, y_train, alpha=RIDGE_ALPHA)

    # Predict on Test Data
    y_pred = predict_hdc_regression(model, X_test_encoded)

    # Evaluate Model Performance
    evaluate_model(y_test, y_pred)

    # Plot Actual vs Predicted Values
    plot_results(y_test, y_pred)

    # Print Sample Predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(y_test))):
        print(f"Actual: {y_test[i]:.4f}, Predicted: {y_pred[i]:.4f}")

if __name__ == '__main__':
    main()