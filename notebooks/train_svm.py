import numpy as np
from sklearn.model_selection import train_test_split

# Load data
X = np.load("features.npy")
y = np.load("labels.npy")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)
