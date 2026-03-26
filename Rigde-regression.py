import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ===== Load dataset =====
columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

data = pd.read_csv(
    r"C:\Users\khang\OneDrive\Desktop\DAP391m\Auto+mpg\auto-mpg.data",
    delim_whitespace=True,
    names=columns,
    na_values="?"
)

data = data.dropna()

# ===== Select feature & target =====
X = data[["horsepower"]].values
y = data["mpg"].values

# ===== Scale feature =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== Ridge Regression =====
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# ===== Create regression line =====
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_line_scaled = scaler.transform(X_line)
y_line = ridge.predict(X_line_scaled)

# ===== Plot =====
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.6, label="Actual data")
plt.plot(X_line, y_line, label="Ridge Regression")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Ridge Regression between Horsepower and MPG")
plt.legend()
plt.grid(True)
plt.show()
