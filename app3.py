import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data
data = {
    'Age': np.random.randint(20, 80, size=50),
    'Sex': np.random.randint(0, 2, size=50),
    'Cholesterol': np.random.randint(150, 300, size=50),
    'Blood Pressure': np.random.randint(80, 180, size=50),
    'Heart Disease': np.random.randint(0, 2, size=50)
}

# Create DataFrame
df = pd.DataFrame(data)

# Directory path where the file will be saved
directory = '/content'

# Create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Save to CSV
csv_file_path = os.path.join(directory, 'heart_disease_prediction.csv')
df.to_csv(csv_file_path, index=False)

print(f"Dataset created and saved to {csv_file_path}")


from responsibleai import RAIInsights, FeatureMetadata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data
data = {
    'Age': np.random.randint(20, 80, size=50),
    'Sex': np.random.randint(0, 2, size=50),  # 0 for female, 1 for male
    'Cholesterol': np.random.randint(150, 300, size=50),
    'Blood Pressure': np.random.randint(80, 180, size=50),
    'Heart Disease': np.random.randint(0, 2, size=50)  # 0 for no, 1 for yes
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file_path = '/content/heart_disease_prediction.csv'
df.to_csv(csv_file_path, index=False)

print(f"Dataset created and saved to {csv_file_path}")

# Load the synthetic dataset
data = pd.read_csv(csv_file_path)

# Prepare the dataset
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# Ensure 'Heart Disease' column is numeric
y = y.astype(int)

# Identify categorical columns
categorical_features = ['Sex']

# Convert categorical columns to string if they are not already
X[categorical_features] = X[categorical_features].astype(str)

# Check and handle missing values
if X.isnull().any().any() or y.isnull().any():
    X = X.dropna()
    y = y[X.index]  # Align y with the cleaned X

# Verify that there are no NA values after cleaning
if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Data contains NA values after handling. Please check the data cleaning steps.")

# Create a column transformer to preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Ensure no missing values in train and test sets
print("Missing values in X_train:\n", X_train.isnull().sum())
print("Missing values in y_train:\n", y_train.isnull().sum())
print("Missing values in X_test:\n", X_test.isnull().sum())
print("Missing values in y_test:\n", y_test.isnull().sum())

# Concatenate X and y for train and test sets
train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

# Check for missing values after concatenation
print("Missing values in train_df after concatenation:\n", train_df.isnull().sum())
print("Missing values in test_df after concatenation:\n", test_df.isnull().sum())

# Drop rows with missing values if any
train_df = train_df.dropna()
test_df = test_df.dropna()

# Verify no NA values exist after dropping
if train_df.isnull().any().any() or test_df.isnull().any().any():
    raise ValueError("NA values found even after dropping missing values.")

# Convert to AIF360 dataset format
train_data_aif360 = BinaryLabelDataset(
    df=train_df,
    label_names=['Heart Disease'],
    protected_attribute_names=['Sex']
)

test_data_aif360 = BinaryLabelDataset(
    df=test_df,
    label_names=['Heart Disease'],
    protected_attribute_names=['Sex']
)

# Predict using the pipeline
y_pred = pipeline.predict(X_test)

# Convert y_pred into a DataFrame
y_pred_df = pd.DataFrame(data={'Heart Disease': y_pred})

# Combine X_test and y_pred for the predicted dataset
predicted_df = pd.concat([X_test.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)

# Convert the predicted DataFrame to AIF360 BinaryLabelDataset
predicted_data_aif360 = BinaryLabelDataset(
    df=predicted_df,
    label_names=['Heart Disease'],
    protected_attribute_names=['Sex']
)

# Initialize fairness metrics
metric = ClassificationMetric(test_data_aif360, predicted_data_aif360,
                               privileged_groups=[{'Sex': 1}], unprivileged_groups=[{'Sex': 0}])

# Print fairness metrics
print("Disparate Impact:")
print(metric.disparate_impact())

print("Statistical Parity Difference:")
print(metric.statistical_parity_difference())

print("Equal Opportunity Difference:")
print(metric.equal_opportunity_difference())

print("Average Odds Difference:")
print(metric.average_odds_difference())

# Create FeatureMetadata instance with only categorical features
feature_metadata = FeatureMetadata(
    categorical_features=categorical_features
)

# Initialize RAIInsights
rai_insights = RAIInsights(
    model=pipeline,
    train=train_df,  # Use cleaned DataFrame
    test=test_df,    # Use cleaned DataFrame
    target_column='Heart Disease',
    task_type='classification',
    feature_metadata=feature_metadata
)

# Add the components you want to include in the insights
rai_insights.explainer.add()
rai_insights.error_analysis.add()

# Increase total_CFs to 10 or more
rai_insights.counterfactual.add(total_CFs=10, desired_class='opposite')

# Specify treatment features for causal analysis
treatment_features = ['Cholesterol', 'Blood Pressure']  # Example treatment features

# Add causal analysis with treatment features
rai_insights.causal.add(treatment_features=treatment_features)

# Compute the insights
rai_insights.compute()

# Get insights
explainer = rai_insights.explainer.get()
error_analysis = rai_insights.error_analysis.get()
counterfactual = rai_insights.counterfactual.get()
causal = rai_insights.causal.get()

# Print insights
print("Explainer Insights:")
print(explainer)

print("Error Analysis Insights:")
print(error_analysis)

print("Counterfactual Insights:")
print(counterfactual)

print("Causal Insights:")
print(causal)
# Check columns of new data
# Assuming 'new_data' is a DataFrame that you want to check
# Define 'new_data' before using it
new_data = pd.DataFrame({
    'Age': [30, 40],
    'Sex': ['1', '0'],  # Ensure categorical features match the encoding
    'Cholesterol': [200, 180],
    'Blood Pressure': [120, 110]
    # Add more columns as needed
})

# Now you can check the columns
print("Columns in new data:", new_data.columns)
print("Columns expected by the model:", preprocessor.get_feature_names_out())


pipeline.fit(X_train, y_train)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
data = pd.read_csv('/content/heart_disease_prediction.csv')
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# Ensure 'Heart Disease' column is numeric
y = y.astype(int)

# Define categorical features
categorical_features = ['Sex']

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline with training data
pipeline.fit(X_train, y_train)
# Example new data for prediction
new_data = pd.DataFrame({
    'Age': [30, 55],  # Example ages
    'Sex': [0, 1],    # Example sex values (0 for female, 1 for male)
    'Cholesterol': [180, 240],  # Example cholesterol values
    'Blood Pressure': [120, 140]  # Example blood pressure values
})

# Ensure categorical columns are of type string
new_data['Sex'] = new_data['Sex'].astype(str)

# Predict using the fitted pipeline
predictions = pipeline.predict(new_data)

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(data={'Heart Disease': predictions})

# Print the predictions
print(predictions_df)
import pandas as pd

def predict_heart_disease(age, sex, cholesterol, blood_pressure):
    # Create DataFrame for new input
    new_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure]
    })
    new_data['Sex'] = new_data['Sex'].astype(str)

    # Predict using the fitted pipeline
    predictions = pipeline.predict(new_data)

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(data={'Heart Disease': predictions})

    return predictions_df

# Get user input
age = int(input("Enter age: "))
sex = input("Enter sex (0 for female, 1 for male): ")
cholesterol = int(input("Enter cholesterol level: "))
blood_pressure = int(input("Enter blood pressure level: "))

# Get prediction
prediction_result = predict_heart_disease(age, sex, cholesterol, blood_pressure)

# Print the predictions
print("Prediction Result:", prediction_result)
