# Housing Price Regression Model
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create dataset from the provided data
data = {
    'Area': [2100, 1600, 2400, 1400, 3000, 1800, 2200, 1700, 2500, 1200],
    'Bedrooms': [3, 2, 4, 2, 5, 3, 3, 2, 4, 2],
    'Bathrooms': [2, 1, 3, 1, 4, 2, 2, 2, 3, 1],
    'Stories': [2, 1, 2, 1, 3, 2, 2, 1, 2, 1],
    'Parking': [2, 1, 2, 0, 3, 1, 2, 1, 2, 0],
    'Mainroad': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No'],
    'Furnishing': ['Furnished', 'Unfurnished', 'Semi-furnished', 'Unfurnished', 'Furnished', 
                   'Semi-furnished', 'Furnished', 'Unfurnished', 'Furnished', 'Unfurnished'],
    'Price': [500000, 300000, 650000, 270000, 850000, 400000, 520000, 320000, 700000, 250000]
}

df = pd.DataFrame(data)

# Handle categorical variables
label_encoder = LabelEncoder()
df['Mainroad'] = label_encoder.fit_transform(df['Mainroad'])

# Handle the Furnishing feature with one-hot encoding
furnishing_dummies = pd.get_dummies(df['Furnishing'], prefix='Furnishing')
df = pd.concat([df, furnishing_dummies], axis=1)
df.drop('Furnishing', axis=1, inplace=True)

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Parking']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Features and target
X = df.drop('Price', axis=1)
y = df['Price']  # Target variable is Price

# Splitting and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize regression models
model_lr = LinearRegression()
model_dtr = DecisionTreeRegressor(max_depth=3)
model_svr = SVR(kernel='linear', C=1.0)

# Train models
model_lr.fit(X_train, y_train)
model_dtr.fit(X_train, y_train)
model_svr.fit(X_train, y_train)

# Evaluation
reg_models = {
    'Linear Regression': model_lr,
    'Decision Tree Regressor': model_dtr,
    'SVR': model_svr
}

print("Model Evaluation Results:")
print("-" * 50)

for name, model in reg_models.items():
    print(f"\nEvaluating: {name}")
    
    y_pred = model.predict(X_test)

    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Prediction on new sample
new_house = pd.DataFrame({
    'Area': [2000],  # Area in sq ft
    'Bedrooms': [3], 
    'Bathrooms': [2],
    'Stories': [2],
    'Parking': [1],
    'Mainroad': [1],  # Yes = 1, No = 0
    'Furnishing_Furnished': [1],
    'Furnishing_Semi-furnished': [0],
    'Furnishing_Unfurnished': [0]
})

# Standardize the numerical features of the new house
new_house[numerical_features] = scaler.transform(new_house[numerical_features])

# Predict using trained models
print("\nPredictions for New House:")
print("-" * 50)
for name, model in reg_models.items():
    prediction = model.predict(new_house)
    print(f"Predicted Price by {name}: ${prediction[0]:.2f}")