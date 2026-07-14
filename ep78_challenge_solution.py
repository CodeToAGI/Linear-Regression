"""
CodeToAGI — Machine Learning From Scratch
Episode 5: Linear Regression Explained (From Scratch + scikit-learn)
Challenge Task — SOLUTION

Task recap (shown in the video at ~13:00):
  House data (sqft -> price):
    650 -> 149,500   800 -> 168,500   1000 -> 203,000   1200 -> 227,500
    1500 -> 276,000  1800 -> 317,000  2100 -> 367,500   2400 -> 409,000

  1) Fit sklearn's LinearRegression to this data (sqft -> price).
  2) BONUS: Implement gradient descent FROM SCRATCH in NumPy to fit the
     same slope and intercept. Standardize sqft first for a stable
     learning rate, then convert your fitted line back to the original
     sqft scale.
  3) BONUS 2: Predict the price of a 2000 sqft house with both methods.
     They should agree almost exactly — that's not a coincidence.

If you solved it a different way and got the same numbers, that's a
valid solution too — there's more than one correct way to write this.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# ── Given data ───────────────────────────────────────────────────────────────
sqft = np.array([650, 800, 1000, 1200, 1500, 1800, 2100, 2400], dtype=float)
price = np.array([149500, 168500, 203000, 227500,
                   276000, 317000, 367500, 409000], dtype=float)

# ── Task 1: fit with scikit-learn ───────────────────────────────────────────
X = sqft.reshape(-1, 1)          # sklearn wants a 2D feature matrix
model = LinearRegression()
model.fit(X, price)

sklearn_slope = model.coef_[0]
sklearn_intercept = model.intercept_
print(f"sklearn slope     : ${sklearn_slope:,.2f} per sqft")
print(f"sklearn intercept : ${sklearn_intercept:,.2f}")
# sklearn slope     : $149.40 per sqft
# sklearn intercept : $50,921.67

# ── Task 2 (BONUS): gradient descent from scratch ───────────────────────────
# Standardize sqft first — raw sqft values (~1000s) make gradient descent
# unstable at a normal learning rate. Fit on the scaled feature, then
# convert the learned line back to the original sqft scale at the end.
mean_x, std_x = sqft.mean(), sqft.std()
x_scaled = (sqft - mean_x) / std_x

m, b = 0.0, 0.0                  # start both parameters at zero
learning_rate = 0.1
epochs = 2000
n = len(sqft)

for _ in range(epochs):
    y_pred = m * x_scaled + b
    error = y_pred - price
    grad_m = (2 / n) * np.sum(error * x_scaled)   # dLoss/dm
    grad_b = (2 / n) * np.sum(error)              # dLoss/db
    m -= learning_rate * grad_m
    b -= learning_rate * grad_b

# Convert the scaled-feature line back to the original sqft scale:
#   y = m * (x - mean)/std + b  =  (m/std)*x + (b - m*mean/std)
gd_slope = m / std_x
gd_intercept = b - m * mean_x / std_x
print(f"\ngradient descent slope     : ${gd_slope:,.2f} per sqft")
print(f"gradient descent intercept : ${gd_intercept:,.2f}")
# gradient descent slope     : $149.40 per sqft
# gradient descent intercept : $50,921.67

# ── Task 3 (BONUS): predict a 2000 sqft house, both ways ───────────────────
pred_sklearn = model.predict([[2000]])[0]

x_new_scaled = (2000 - mean_x) / std_x
pred_gd = m * x_new_scaled + b

print(f"\nsklearn predicts 2000 sqft         : ${pred_sklearn:,.2f}")
print(f"gradient descent predicts 2000 sqft: ${pred_gd:,.2f}")
# sklearn predicts 2000 sqft         : $349,721.08
# gradient descent predicts 2000 sqft: $349,721.08

# ── Why this works ───────────────────────────────────────────────────────────
# sklearn's LinearRegression solves for the exact minimum of the Mean
# Squared Error in one shot (a closed-form solution). Gradient descent
# gets there step by step, nudging m and b downhill on the same error
# surface, epoch after epoch. Different paths, same destination — because
# both are minimizing the exact same cost function.

# ── Checks ───────────────────────────────────────────────────────────────────
assert round(sklearn_slope, 2) == 149.40
assert round(sklearn_intercept, 2) == 50921.67
assert abs(gd_slope - sklearn_slope) < 0.01
assert abs(gd_intercept - sklearn_intercept) < 0.01
assert abs(pred_sklearn - pred_gd) < 1.0

print("\nAll checks passed — gradient descent, step by step, lands in the")
print("exact same place as sklearn's one-shot solution.")
