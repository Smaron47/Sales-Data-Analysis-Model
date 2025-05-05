import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the transformed data
data = pd.read_csv('transformed_sales_data.csv')

# Define target variable
data['Target'] = (data['TotalClientSales'] > data['TotalClientSales'].median()).astype(int)

# Define feature sets for each functionality
feature_sets = {
    "Customer Engagement & Retention": ['ClientCode', 'TotalClientSales', 'ClientOrderCount', 'LastPurchaseDate', 'EngagementScore'],
    "Lead Generation & Management": ['LeadSource', 'LeadStatus', 'LeadAge', 'TotalClientSales', 'ConversionRate'],
    "Stock Management & Sales Targeting": ['ProductID', 'StockLevel', 'RestockAlertThreshold', 'SalesTarget', 'TotalProductSales'],
    "Forecasting & Strategy": ['HistoricalSalesData', 'SeasonalityIndex', 'MarketTrends', 'TotalClientSales', 'SalesGrowthRate'],
    "Lead Prioritization": ['LeadScore', 'TotalClientSales', 'LeadAge', 'ConversionProbability', 'ClientEngagementScore']
}

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}

for feature_set_name, feature_set in feature_sets.items():
    X = data[feature_set]
    y = data['Target']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle categorical features using OneHotEncoder
    categorical_features = ['ClientCode', 'LeadSource'] # Add other categorical columns if needed

    # Create a OneHotEncoder object
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # sparse=False for dense output

    # Fit the encoder on the training data
    X_train_encoded = encoder.fit_transform(X_train[categorical_features])
    X_test_encoded = encoder.transform(X_test[categorical_features])

    # Get feature names after encoding
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    # Create DataFrames for encoded features
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_feature_names, index=X_test.index)

    # Drop original categorical columns and concatenate encoded features
    X_train = X_train.drop(columns=categorical_features, errors='ignore')
    X_train = pd.concat([X_train, X_train_encoded_df], axis=1)

    X_test = X_test.drop(columns=categorical_features, errors='ignore')
    X_test = pd.concat([X_test, X_test_encoded_df], axis=1)

    # Standardize numerical features
    numerical_features = X_train.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])



# Print results in a formatted way
print("Model Accuracy Comparison:\n")
print(f"{'Model Name':<25} {'Feature Set':<50} {'Accuracy':<10}")
print("="*85)
for (model_name, feature_set_name), accuracy in results.items():
    print(f"{model_name:<25} {feature_set_name:<50} {accuracy:.4f}")

# Determine the best model
best_model = max(results, key=results.get)
print("\nBest Model:")
print(f"Model Name: {best_model[0]}")
print(f"Feature Set: {best_model[1]}")
print(f"Accuracy: {results[best_model]:.4f}")
