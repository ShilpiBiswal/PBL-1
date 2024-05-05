import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from collections import defaultdict

#preprocess new data
def preprocess_data(df, label_encoders, scaler):
    for feat, le in label_encoders.items():
        if feat in df.columns:
            common_label = le.classes_[0]  #default
            df[feat] = df[feat].astype(str).apply(lambda x: x if x in le.classes_ else common_label)
            df[feat] = le.transform(df[feat])


    scaled_df = scaler.transform(df)

    return scaled_df

#function to evaluate random forest
def evaluate_rf(x_scaled, y):

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)


    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


    results = defaultdict(dict)

    for i, (train_index, test_index) in enumerate(skf.split(x_scaled, y)):
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf_classifier.fit(x_train, y_train)

        y_pred = rf_classifier.predict(x_test)

        results[f'Split {i + 1}'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }

    #average accuracy
    avg_accuracy = np.mean([result['accuracy'] for result in results.values()])

    #average confusion matrix
    avg_confusion_matrix = sum(result['confusion_matrix'] for result in results.values()) / len(results)

    #average classification report
    avg_classification_report = defaultdict(lambda: defaultdict(list))  # Default to avoid 'KeyError'

    for result in results.values():
        report = result['classification_report']
        for key, metrics in report.items():
            if isinstance(metrics, dict):  # Collecting 'precision', 'recall', 'f1-score', etc.
                for metric, value in metrics.items():
                    avg_classification_report[key][metric].append(value)

    #average for each metric
    for key, metrics in avg_classification_report.items():
        for metric, values in metrics.items():
            avg_classification_report[key][metric] = sum(values) / len(values)

    return {
        'avg_accuracy': avg_accuracy,
        'avg_confusion_matrix': avg_confusion_matrix,
        'avg_classification_report': avg_classification_report,
    }


data_set = pd.read_csv('Sleep_health_and_lifestyle_dataset (2).csv')
data_set_revised = data_set.drop(columns=['Person ID'])


data_set_revised[['SYS', 'DIA']] = data_set_revised['Blood Pressure'].str.split("/", expand=True).apply(pd.to_numeric)
data_set_revised.drop('Blood Pressure', axis=1, inplace=True)


cat_feats = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
label_encoders = {}

for feat in cat_feats:
    if data_set_revised[feat].dtype == 'object':
        label_encoders[feat] = LabelEncoder()
        data_set_revised[feat] = label_encoders[feat].fit_transform(data_set_revised[feat])


data_set_revised.dropna(inplace=True)


x = data_set_revised.drop(columns=['Sleep Disorder'])
y = data_set_revised['Sleep Disorder']


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


rf_results = evaluate_rf(x_scaled, y)

print("Cross-Validation Results:")
print(f"\nAverage Accuracy: {rf_results['avg_accuracy']:.2f}")
print("Average Confusion Matrix:")
print(rf_results['avg_confusion_matrix'])
print("Average Classification Report:")
print(pd.DataFrame(rf_results['avg_classification_report']).T)


patient_data_2 = pd.DataFrame({
    'Gender': [1],
    'Age': [28],
    'Occupation': [6],
    'Sleep Duration': [5.9],
    'Quality of Sleep': [4],
    'Physical Activity Level': [30],
    'Stress Level': [8],
    'BMI Category': [2],
    'Heart Rate': [85],
    'Daily Steps': [3000],
    'SYS': [140],
    'DIA': [90]
})


processed_patient_data = preprocess_data(patient_data_2, label_encoders, scaler)
prediction_result = rf_classifier.predict(processed_patient_data)


prediction_label = 'No Apnea' if prediction_result[0] == 2 else 'Has Apnea'

print(f"\nPrediction for the given patient: {prediction_label}")
