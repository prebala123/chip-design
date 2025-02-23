import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

from sklearn.metrics import accuracy_score

# Hyperparameters
DIMENSION = 10000  # High-dimensional space
PERTURBATION_SCALE = 0.0005  # Reduced perturbation
LEARNING_RATE = 0.01  # Gradual reinforcement rate

def random_hv():
    return np.random.choice([-1, 1], DIMENSION)

def structured_hv(value, min_val, max_val):
    # Structured encoding based on intensity
    threshold =(value - min_val) / (max_val - min_val)
    return np.where(np.random.rand(DIMENSION) < threshold, 1, -1)

def bind(vec1, vec2):
    return vec1 * vec2

def bundle(vecs):
    return np.sign(np.sum(vecs, axis=0))

def similarity(vec1, vec2):
    return np.dot(vec1, vec2)

def adaptive_redistribute(vec, feedback):
    # Controlled redistribution based on feedback
    perturbation = np.random.choice([-1, 1], DIMENSION) * PERTURBATION_SCALE * max(0, (1 - feedback))
    return np.sign(vec + perturbation)

def reinforce(memory_vec, input_vec, correct):
    # Gradual reinforcement with learning rate
    adjustment = LEARNING_RATE * (input_vec if correct else -input_vec)
    return np.sign(memory_vec + adjustment)


import numpy as np
from sklearn.model_selection import train_test_split

# Sample data matrix (X) and labels (y)
X = node_features['instance_features']
y = targets['classify'].numpy().flatten()

# Split data and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Structured feature encoding
min_vals = X_train.min(axis=0)
max_vals = X_train.max(axis=0)
feature_hvs = [random_hv() for _ in range(X_train.shape[1])]
class_hvs = {label: random_hv() for label in np.unique(y_train)}

# Training 
memory_vectors = {label: np.zeros(DIMENSION) for label in class_hvs}
for x, y in zip(X_train, y_train):
    encoded_features = [bind(feature_hvs[i], structured_hv(feature, min_vals[i], max_vals[i])) for i, feature in enumerate(x)]
    bundled_vector = bundle(encoded_features)
    feedback = similarity(bundled_vector, memory_vectors[y]) / DIMENSION
    bundled_vector = adaptive_redistribute(bundled_vector, feedback)
    memory_vectors[y] = reinforce(memory_vectors[y], bundled_vector, True)

# Normalize memory vectors
for label in memory_vectors:
    memory_vectors[label] = np.sign(memory_vectors[label])

# Testing
predictions = []
for x, true_label in zip(X_test, y_test):
    encoded_features = [bind(feature_hvs[i], structured_hv(feature, min_vals[i], max_vals[i])) for i, feature in enumerate(x)]
    bundled_vector = bundle(encoded_features)
    sims = {label: similarity(bundled_vector, mem_vec) for label, mem_vec in memory_vectors.items()}
    predicted_label = max(sims, key=sims.get)
    predictions.append(predicted_label)
    # Gradual reinforcement for correct predictions
    memory_vectors[true_label] = reinforce(memory_vectors[true_label], bundled_vector, predicted_label == true_label)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy with only node_features['instance_features'] from superblue14 with structured encoding and gradual reinforcement: {accuracy:.4f}")