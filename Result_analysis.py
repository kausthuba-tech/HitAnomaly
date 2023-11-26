#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# Results for HitAnomaly, SVM, and OpenAI models
hit_anomaly_results = {
    'Accuracy': 0.92,
    'Precision': 0.16,
    'Recall': 0.27,
    'F1 Score': 0.21
}

svm_results = {
    'Accuracy': 0.25,
    'Precision': 0.019,
    'Recall': 0.61,
    'F1 Score': 0.039
}

# Assuming you have OpenAI results
openai_results = {
    'Accuracy': 0.86,
    'Precision': 0.54,
    'Recall': 0.54,
    'F1 Score': 0.54
}

# List of metrics
metrics = list(hit_anomaly_results.keys())

# Results for each metric
hit_anomaly_values = [hit_anomaly_results[metric] for metric in metrics]
svm_values = [svm_results[metric] for metric in metrics]
openai_values = [openai_results[metric] for metric in metrics]

# Set up bar positions
bar_width = 0.25
index = np.arange(len(metrics))

# Create bar graph
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(index - bar_width, hit_anomaly_values, bar_width, label='HitAnomaly')
bar2 = ax.bar(index, svm_values, bar_width, label='SVM')
bar3 = ax.bar(index + bar_width, openai_values, bar_width, label='OpenAI')

# Labeling and customization
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Anomaly Detection Models')
ax.set_xticks(index)
ax.set_xticklabels(metrics)
ax.legend()

# Show the plot
plt.show()


# In[ ]:





# In[ ]:




