# model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)  # shape (150,1)

# One-hot encode targets
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build ANN
model = Sequential([
    Dense(10, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {acc*100:.2f}%")

# Save model, encoder, and scaler
model.save("iris_ann_model.h5")
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
np.save("encoder_categories.npy", encoder.categories_[0])
print("Model, scaler, and encoder saved successfully.")
