import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from scipy.stats import mode

df = pd.read_csv(r"D:\course\P\Project4\pythonProject\2\Advertising Budget and Sales.csv")
print(df.columns)
X = df['TV Ad Budget ($)'] # 广告预算作为特征
y = df['Sales ($)']  # 销售额作为目标

np.random.seed(0)
# Split the dataset into train (80%) and test (20%)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


noise_level = 0.0 # Adjust the noise level (0 to 0.2)
y_train_noisy = y_train + np.random.normal(0, noise_level, y_train.shape)

Q1_X, Q3_X = np.percentile(X_train, [25, 75])
IQR_X = Q3_X - Q1_X

Q1_y, Q3_y = np.percentile(y_train, [25, 75])
IQR_y = Q3_y - Q1_y


outlier_X_high = Q3_X + 1.5 * IQR_X
outlier_X_low = Q1_X - 1.5 * IQR_X

outlier_y_high = Q3_y + 1.5 * IQR_y
outlier_y_low = Q1_y - 1.5 * IQR_y


outlier_fraction = 0.3
num_outliers = int(outlier_fraction * len(X_train) / 2)


outliers_X_high = np.random.uniform(outlier_X_high, outlier_X_high + 1, num_outliers)
outliers_y_high = np.random.uniform(outlier_y_high, outlier_y_high + 1, num_outliers)


outliers_X_low = np.random.uniform(outlier_X_low - 1, outlier_X_low, num_outliers)
outliers_y_low = np.random.uniform(outlier_y_low - 1, outlier_y_low, num_outliers)


outliers_X_sides = np.random.uniform(X_train.min(), X_train.max(), num_outliers * 2)
outliers_y_sides = np.random.uniform(y_train.min(), y_train.max(), num_outliers * 2)


outliers_X = np.concatenate([outliers_X_high, outliers_X_low, outliers_X_sides])
outliers_y = np.concatenate([outliers_y_high, outliers_y_low, outliers_y_sides])


X_train_with_outliers = np.concatenate([X_train, outliers_X])
y_train_with_outliers = np.concatenate([y_train_noisy, outliers_y])


shuffled_indices = np.random.permutation(len(X_train_with_outliers))
X_train_shuffled = X_train_with_outliers[shuffled_indices].reshape(-1, 1)
y_train_shuffled = y_train_with_outliers[shuffled_indices].reshape(-1, 1)


sorted_indices = np.argsort(X_train_shuffled, axis=0).flatten()
X_train_sorted = X_train_shuffled[sorted_indices]
y_train_sorted = y_train_shuffled[sorted_indices]


Q1_y = np.percentile(y_train_sorted, 25)
median_y = np.percentile(y_train_sorted, 50)
Q3_y = np.percentile(y_train_sorted, 75)


def find_closest_all(y_values, target):
    distances = np.abs(y_values - target)
    min_distance = np.min(distances)
    closest_indices = np.where(distances == min_distance)[0]
    return closest_indices

closest_to_Q1 = find_closest_all(y_train_sorted, Q1_y)
closest_to_median = find_closest_all(y_train_sorted, median_y)
closest_to_Q3 = find_closest_all(y_train_sorted, Q3_y)
#nprint(closest_to_Q1, closest_to_median, closest_to_Q3)


x1 = X_train_sorted[closest_to_Q1]
x2 = X_train_sorted[closest_to_median]
x3 = X_train_sorted[closest_to_Q3]
# print(x1, x2, x3)

x1_mode = mode(x1).mode[0]
x2_mode = mode(x2).mode[0]
x3_mode = mode(x3).mode[0]
# Step 4: Fit lines to each pair of points
def fit_line(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

# Fit lines for each pair
# m1, b1 = fit_line(x1_mode, Q1_y, x2_mode, median_y)
m2, b2 = fit_line(x1_mode, Q1_y, x3_mode, Q3_y)
# m3, b3 = fit_line(x2_mode, median_y, x3_mode, Q3_y)


def average_lines(m1, b1, m2, b2, m3, b3):
    m_avg = (m1 + m2 + m3) / 3
    b_avg = (b1 + b2 + b3) / 3
    return m_avg, b_avg

# m_avg, b_avg = average_lines(m1, b1, m2, b2, m3, b3)


# Make predictions on the test set
# y_pred1 = m1 * X_test + b1
y_pred2 = m2 * X_test + b2
#y_pred3 = m3 * X_test + b3
#y_pred_avg = m_avg * X_test + b_avg

# Calculate accuracy for each line and the average line
#mse1 = mean_squared_error(y_test, y_pred1)
#mse2 = mean_squared_error(y_test, y_pred2)
#mse3 = mean_squared_error(y_test, y_pred3)
mae2 = mean_absolute_error(y_test, y_pred2)
r_square2 = r2_score(y_test, y_pred2)
#mse_avg = mean_squared_error(y_test, y_pred_avg)
#mae_avg = mean_absolute_error(y_test, y_pred_avg)
#r_square_avg = r2_score(y_test, y_pred_avg)

# Print the accuracy results
# print(f"L1 Mean Squared Error of Line 1: {mse1:.2f}")
# print(f"L1 Mean Squared Error of Line 3: {mse3:.2f}")
# print(f"L1 Mean Squared Error of Line 2: {mse2:.2f}")
print("QBR Mean average Error of Line 2: ", mae2)
print("QBR R-squared (R²) of Line 2:", r_square2)
# print(f"L1 Mean Squared Error(MSE) of Averaged Line: ", mse_avg)
# print("L1 Mean Absolute Error(MAE) of Averaged Line:", mae_avg)
# print("L1 R-squared (R²) of Averaged Line:", r_square_avg)



X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
# Instantiate the LinearRegression model (this is the basic OLS regression)
model = LinearRegression()

# Fit the model to your training data
model.fit(X_train_shuffled, y_train_shuffled)

# Predict on the test set
y_pred = model.predict(X_test)

# Print the coefficients and intercept
# print(f"Coefficients: {model.coef_}")
# print(f"Intercept: {model.intercept_}")

# Calculate the Mean Squared Error (MSE)
L2_mse = mean_squared_error(y_test, y_pred)
L2_mae = mean_absolute_error(y_test, y_pred)
L2_r_square = r2_score(y_test, y_pred)
# print(f"Mean Squared Error: ", L2_mse)
print("Mean Absolute Error (MAE) for Linear Regression:", L2_mae)
print("R-squared (R²) for Linear Regression:", L2_r_square)

# Plot the noisy data, outliers, and regression lines
plt.figure(figsize=(10, 6))

# Plot noisy data and outliers
plt.scatter(X_train_sorted, y_train_sorted, label="Data", color='blue', alpha=0.5)
plt.scatter(outliers_X, outliers_y, label="Outliers", color='pink', alpha=0.5)

# Plot individual lines and averaged line on the same plot
plt.plot(X, m2 * X + b2, label='Quantile-based Regression', linestyle='-', color='black')
plt.plot(X_test, y_pred, label='Linear Regression', linestyle='-', color='green')

# Highlight (x1, Q1_y), (x2, median_y), (x3, Q3_y) points
plt.scatter(x1_mode, Q1_y, color='red', s=100, label="Q1 Point")
plt.scatter(x2_mode, median_y, color='brown', s=100, label="Median Point")
plt.scatter(x3_mode, Q3_y, color='red', s=100, label="Q3 Point")

# Annotate each of the points
plt.annotate("Q1", (x1_mode, Q1_y), textcoords="offset points", xytext=(-10, 10), ha='center', color='red')
plt.annotate("Median", (x2_mode, median_y), textcoords="offset points", xytext=(10, -10), ha='left', color='brown')
plt.annotate("Q3", (x3_mode, Q3_y), textcoords="offset points", xytext=(-10, 10), ha='center', color='red')

# Plot settings
plt.xlabel('X')
plt.ylabel('y')
plt.title(f"Data with Outliers (fraction={outlier_fraction}), and QBR")
plt.legend()
plt.show()
