# Part 1: Data Preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv('customer_segmentation_dataset.csv', sep=',')


# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Spending Score'].fillna(df['Spending Score'].mean(), inplace=True)
df['Male'].fillna(df['Male'].mode()[0], inplace=True)




# Encode categorical variables
label = LabelEncoder()
df['Gender'] = label.fit_transform(df['Gender'])
df['Region'] = label.fit_transform(df['Region'])
df['Customer Segment'] = label.fit_transform(df['Customer Segment'])

# Standardize numerical features
scalar = StandardScaler()
df[['Age', 'Income', 'Spending Score']] = scalar.fit_transform(df[['Age', 'Income', 'Spending Score']])

# Segregate features and target
x = df[['Customer ID', 'Age', 'Income', 'Spending Score', 'Gender', 'Region']]
y = df['Customer Segment']





# Part 2: Data Splitting and Model Training


# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Initialize models
model_lr = LogisticRegression()
model_svc = SVC()
model_dt = DecisionTreeClassifier()



# Tuned models
t_model_lr = LogisticRegression(C=0.3, solver='liblinear')
t_model_dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
t_model_svc = SVC(C=0.5, kernel='linear', gamma='scale')

# Train models
t_model_lr.fit(x_train, y_train)
t_model_svc.fit(x_train, y_train)
t_model_dt.fit(x_train, y_train)

# Part 3: Model Evaluation


models = {
    'Logistic Regression': t_model_lr,
    'SVC': t_model_svc,
    'Decision Tree': t_model_dt
}

for name, model in models.items():
    print(f"\nEvaluating: {name}")
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    print("Training Performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(classification_report(y_train, y_pred_train))

    print("Testing Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))

    # K-Fold Cross Validation
    cv_scores = cross_val_score(model, x, y, cv=5)
    print(f"Average K-Fold Score (5 folds): {cv_scores.mean():.4f}")

    # kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # scores = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')


# Prediction on new data
new_data = pd.DataFrame({
    'Customer ID': [1111],
    'Age': [46],
    'Income': [61900],
    'Spending Score': [30],
    'Gender': [1],
    'Region': [1]
})

# Preprocess new data
new_data[['Age', 'Income', 'Spending Score']] = scalar.transform(new_data[['Age', 'Income', 'Spending Score']])


# Predict using trained models
for name, model in models.items():
    result = model.predict(new_data)
    print(f"Prediction by {name}: Customer belongs to Segment {result[0]}")
