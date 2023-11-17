import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix

# Load HDFS log dataset
log_dataset = pd.read_csv("/Users/kausthubavanam/Downloads/Archive/HDFS_2k.log_structured (1).csv")

# Preprocess the data
log_dataset['BlockId'] = log_dataset['Content'].apply(lambda x: re.search(r'\bblk_(-|)\d+\b', str(x)).group(0))
anomaly_label = pd.read_csv("anomaly_label.csv")
new_log_dataset = log_dataset.merge(anomaly_label, on='BlockId', how='left')

# Map labels to 1 (Anomaly) and 0 (Normal)
labels = {'Normal': 0, 'Anomaly': 1}
new_log_dataset['Label'] = new_log_dataset['Label'].map(labels)

# Select relevant columns
selected_columns = ['Component', 'Content', 'Label']
new_log_dataset = new_log_dataset[selected_columns]

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    new_log_dataset['Content'], new_log_dataset['Label'], test_size=0.2, random_state=42
)

# Convert logs to TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Train One-Class SVM model
svm_model = OneClassSVM(kernel='linear', gamma='auto')
svm_model.fit(train_features)

# Predict anomalies on the test set
predictions = svm_model.predict(test_features)

threshold = 0.5  # You can adjust this threshold
predictions = (predictions > threshold).astype(int)
# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)

# Calculate precision, recall, and f1-score
true_positives = conf_matrix[1, 1]
false_positives = conf_matrix[0, 1]
false_negatives = conf_matrix[1, 0]

precision = true_positives/ (true_positives + false_positives + 1e-9)
recall = true_positives+1/ (true_positives + false_negatives + 1e-9)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)