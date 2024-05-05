import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from collections import defaultdict

''' Function to preprocess data
def preprocess_data(df, label_encoders, scaler):
    for feat, le in label_encoders.items():
        if feat in df.columns:
            common_label = le.classes_[0]  # Default for unknown categories
            df[feat] = df[feat].astype(str).apply(lambda x: x if x in le.classes_ else common_label)
            df[feat] = le.transform(df[feat])

    scaled_df = scaler.transform(df)
    return scaled_df'''

# Function to evaluate Random Forest with Stratified K-Fold cross-validation
def evaluate_rf(x_scaled, y):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    results = defaultdict(dict)

    # Stratified K-Fold cross-validation
    for i, (train_index, test_index) in enumerate(skf.split(x_scaled, y)):
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf_classifier.fit(x_train, y_train)  # Train the classifier
        y_pred = rf_classifier.predict(x_test)  # Predict the test set

        # Store evaluation metrics
        results[f'Split {i + 1}'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }

    avg_accuracy = np.mean([result['accuracy'] for result in results.values()])
    avg_confusion_matrix = sum(result['confusion_matrix'] for result in results.values()) / len(results)
    avg_classification_report = defaultdict(lambda: defaultdict(list))

    for result in results.values():
        report = result['classification_report']
        for key, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    avg_classification_report[key][metric].append(value)

    for key, metrics in avg_classification_report.items():
        for metric, values in metrics.items():
            avg_classification_report[key][metric] = sum(values) / len(values)

    # Return both the evaluation results and the trained classifier
    return {
        'avg_accuracy': avg_accuracy,
        'avg_confusion_matrix': avg_confusion_matrix,
        'avg_classification_report': avg_classification_report,
    }, rf_classifier


# Function to print encoded values from LabelEncoders
def print_encoded_values(label_encoders):
    for feature, le in label_encoders.items():
        print(f"Encoded values for {feature}:")
        for original, encoded in zip(le.classes_, range(len(le.classes_))):
            print(f"  {original} => {encoded}")

# Load the dataset and preprocess it
data_set = pd.read_csv('Sleep_health_and_lifestyle_dataset (2).csv')
data_set_revised = data_set.drop(columns=['Person ID'])  # Remove identifier

# Split blood pressure into SYS and DIA, remove original 'Blood Pressure'
data_set_revised[['SYS', 'DIA']] = data_set_revised['Blood Pressure'].str.split("/", expand=True).apply(pd.to_numeric)
data_set_revised.drop('Blood Pressure', axis=1, inplace=True)

# Label encoding for categorical features
cat_feats = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
label_encoders = {}

for feat in cat_feats:
    if data_set_revised[feat].dtype == 'object':
        label_encoders[feat] = LabelEncoder()
        data_set_revised[feat] = label_encoders[feat].fit_transform(data_set_revised[feat])

# Remove rows with missing data
data_set_revised.dropna(inplace=True)

# Separate features and target variable
x = data_set_revised.drop(columns=['Sleep Disorder'])
y = data_set_revised['Sleep Disorder']

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Get evaluation results and the trained classifier
rf_results, rf_classifier = evaluate_rf(x_scaled, y)

# Print encoded values
print("Encoded values:")
print_encoded_values(label_encoders)

# Now, access elements from `rf_results`
print("Cross-Validation Results:")
print(f"\nAverage Accuracy: {rf_results['avg_accuracy']:.2f}")
print("Average Confusion Matrix:")
print(rf_results['avg_confusion_matrix'])
print("Average Classification Report:")
print(pd.DataFrame(rf_results['avg_classification_report']).T)

# Create a new patient data instance for prediction
patient_data_1 = pd.DataFrame({
    'Gender': [1],
    'Age': [28],
    'Occupation': [1],
    'Sleep Duration': [6.2],
    'Quality of Sleep': [6],
    'Physical Activity Level': [60],
    'Stress Level': [8],
    'BMI Category': [0],
    'Heart Rate': [75],
    'Daily Steps': [10000],
    'SYS': [125],
    'DIA': [80]
})
patient_data_2 = pd.DataFrame({
    'Gender': [1],
    'Age' : [28],
    'Occupation' : [6],
    'Sleep Duration': [5.9],
    'Quality of Sleep' : [4],
    'Physical Activity Level' : [30],
    'Stress Level' : [8],
    'BMI Category' : [2],
    'Heart Rate' : [85],
    'Daily Steps' : [3000],
    'SYS' : [140],
    'DIA' : [90]
})

scaled_patient_data_1 = scaler.transform(patient_data_1)
scaled_patient_data_2 = scaler.transform(patient_data_2)

# Predict using the trained Random Forest classifier

patient_predictions_1 = rf_classifier.predict(scaled_patient_data_1)
patient_predictions_2 = rf_classifier.predict(scaled_patient_data_2)
print(patient_predictions_1)
print(patient_predictions_2)

print("\nPredictions for patient data:")
if(patient_predictions_1==1):
    print("Patient has been diagnosed with Sleep Apnea")

else:
    print('Patient does not have sleep apnea')

if(patient_predictions_2==1):
  print("Patient has been diagnosed with Sleep Apnea")
else:
    print('Patient does not have sleep apnea')
