import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv(r"C:\Users\MUZZAMMIL\Downloads\student-mat.csv", sep=";")

print("Dataset shape:", data.shape)
print(data.head())

# ==========================================
# 2. CREATE TARGET
# ==========================================
# G3 is the student's final grade.
# G3 >= 10  -> Pass (1)
# G3 < 10   -> Fail (0)

data["pass"] = (data["G3"] >= 10).astype(int)

# ==========================================
# 3. SELECTING NUMERICAL FEATURES
# ==========================================

features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Walc",
    "health",
    "absences"
]

X = data[features].values
y = data["pass"].values.reshape(-1, 1)

# ==========================================
# 4. CHECKING CORRELATION
# ==========================================

correlations = data[features + ["G3"]].corr()["G3"]

print("\nCorrelation with final grade (G3):")
print(correlations.sort_values(ascending=False))

# ==========================================
# 5. NORMALIZING FEATURES
# ==========================================

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X = (X - mean) / std

# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

np.random.seed(42)

indices = np.random.permutation(len(X))

train_size = 276

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 7. ADD BIAS TERM
# ==========================================

X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# ==========================================
# 8. SIGMOID FUNCTION
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ==========================================
# 9. LOGISTIC REGRESSION
# ==========================================

weights = np.zeros((X_train_bias.shape[1], 1))

learning_rate = 0.01
iterations = 10000

losses = []

for i in range(iterations):

    # Prediction
    z = X_train_bias @ weights
    predictions = sigmoid(z)

    # Binary cross-entropy loss
    epsilon = 1e-8

    loss = -np.mean(
        y_train * np.log(predictions + epsilon)
        + (1 - y_train) * np.log(1 - predictions + epsilon)
    )

    losses.append(loss)

    # Gradient
    gradient = (
        X_train_bias.T @ (predictions - y_train)
        / len(X_train_bias)
    )

    # Update weights
    weights = weights - learning_rate * gradient

# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

train_probabilities = sigmoid(X_train_bias @ weights)
test_probabilities = sigmoid(X_test_bias @ weights)

train_predictions = (train_probabilities >= 0.5).astype(int)
test_predictions = (test_probabilities >= 0.5).astype(int)

# ==========================================
# 11. ACCURACY
# ==========================================

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print("\nLogistic Regression Results")
print("---------------------------")
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)

# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

true_positive = np.sum(
    (test_predictions == 1) & (y_test == 1)
)

true_negative = np.sum(
    (test_predictions == 0) & (y_test == 0)
)

false_positive = np.sum(
    (test_predictions == 1) & (y_test == 0)
)

false_negative = np.sum(
    (test_predictions == 0) & (y_test == 1)
)

print("\nConfusion Matrix")
print("----------------")
print("True Positive :", true_positive)
print("True Negative :", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)

# ==========================================
# 13. PRECISION / RECALL
# ==========================================

precision = true_positive / (true_positive + false_positive + 1e-8)

recall = true_positive / (true_positive + false_negative + 1e-8)

f1_score = (
    2 * precision * recall
    / (precision + recall + 1e-8)
)

print("\nEvaluation")
print("----------")
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1_score)

# ==========================================
# 14. PLOT TRAINING LOSS
# ==========================================

plt.plot(losses)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression Training Loss")

plt.show()