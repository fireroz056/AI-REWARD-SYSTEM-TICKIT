import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
customer_profile = pd.read_csv('customer_profile.csv')
behavioral_data = pd.read_csv('behavioral_data.csv')
reward_catalogue = pd.read_csv('reward_catalogue.csv')
reward_data = pd.read_csv('reward_data_with_preferences.csv')

# Merge dataframes to create a comprehensive dataset
merged_data = reward_data.merge(customer_profile, on='user_id')
merged_data = merged_data.merge(behavioral_data, on='user_id')
merged_data = merged_data.merge(reward_catalogue, on='reward_id')

# Drop unnecessary columns
merged_data = merged_data.drop(columns=['reward_name', 'reward_category_y', 'points_required'])

# Rename the remaining reward_category column
merged_data = merged_data.rename(columns={'reward_category_x': 'reward_category'})

# Define the features and the target
features = ['user_id', 'reward_id', 'points', 'age', 'gender', 'income', 'location', 
            'login_frequency', 'purchase_frequency', 'average_spent_time_on_site', 
            'average_purchase_value', 'reward_category']
target = 'preferred_reward_category'

# One-hot encode the categorical columns
merged_data = pd.get_dummies(merged_data, columns=['gender', 'location', 'reward_category'], drop_first=True)

# Update the features list to include the new one-hot encoded columns
features = [col for col in merged_data.columns if col != target]

# Split the data into features (X) and target (y)
X = merged_data[features]
y = merged_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'reward_recommendation_model.pkl')

# Generate reward recommendations for new users
new_users = pd.DataFrame({
    'user_id': [101, 102, 103],
    'reward_id': [1, 2, 3],
    'points': [500, 300, 200],
    'age': [25, 34, 45],
    'gender': ['Female', 'Male', 'Female'],
    'income': [55000, 72000, 64000],
    'location': ['New York', 'Los Angeles', 'Chicago'],
    'login_frequency': [50, 60, 40],
    'purchase_frequency': [5, 10, 7],
    'average_spent_time_on_site': [20, 30, 25],
    'average_purchase_value': [200, 300, 250],
    'reward_category': ['Electronics', 'Books', 'Food']
})

# One-hot encode the 'gender', 'location', and 'reward_category' columns for new users
new_users = pd.get_dummies(new_users, columns=['gender', 'location', 'reward_category'], drop_first=True)

# Ensure the new users dataframe has the same columns as the training data
new_users = new_users.reindex(columns=X.columns, fill_value=0)

# Predict the preferred reward category for new users
new_users['predicted_reward_category'] = model.predict(new_users[features])
print(new_users[['user_id', 'predicted_reward_category']])