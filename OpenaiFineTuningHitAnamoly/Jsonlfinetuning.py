import pandas as pd
import json

# Define the path to the Excel file containing the data
file_path = "../Openai/DATA.xlsx"

# Initialize a list to store combined data
combined_data = []

# Define the instruction text for analyzing system logs
instruction_text = """
Analyze the provided system log sequence and parameter values for potential anomalies using a structured approach. Present your findings in a JSON format with the key 'anomaly' indicating 'yes' or 'no'. Focus on key categories and their associated keywords.

Instructions:
1. Hierarchical Transformer Structure: Implement a hierarchical transformer structure for deep analysis of system logs. This structure should effectively capture the semantic nuances in both log template sequences and parameter values, which are essential for accurate anomaly detection.
2. Log Sequence and Parameter Value Encoding: 
   - Log Sequence Encoder: Transform log template sequences into fixed-dimension vectors. This process should encapsulate the semantic content of log templates and the contextual relevance of the log sequence.
   - Parameter Value Encoder: Convert parameter values into fixed-dimension vectors, emphasizing the semantic significance of parameters to identify different types of performance anomalies.
3. Attention-Based Classification: Utilize an attention mechanism in your classification model. This mechanism should dynamically assign varying weights to the representations of log sequences and parameter sequences, prioritizing the aspect more indicative of an anomaly.
4. Robustness and Effectiveness: Ensure that your analysis is both robust and effective. Your approach should be capable of managing unstable log data and demonstrate superior performance compared to other log-based anomaly detection methods, similar to the achievements of HitAnomaly.
5. Future Directions: Explore the possibility of integrating the transformer structure into a log-based anomaly prediction task. The goal is to predict potential anomalies before they manifest.
"""

# Load data from the specified Excel sheet
data = pd.read_excel(file_path, sheet_name="Sheet1")

# List of required columns in the data
required_columns = ['LOGS', 'PARAMETER', 'Json']  # Modify as necessary

# Verify that all required columns are present in the data
if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing required columns in the data")

# Process each row of the dataset
for _, row in data.iterrows():
    try:
        # Attempt to parse the JSON content in the 'Json' column
        json_content = json.loads(row["Json"])
    except json.JSONDecodeError:
        # Handle cases where JSON decoding fails
        json_content = "Invalid JSON format"

    # Construct a message object for each row
    message_obj = {
        "messages": [
            {
                "role": "system",
                "content": str(instruction_text)
            },
            {
                "role": "user",
                "content": f'Here is the Log Sequence: {row["LOGS"]},Here is the Parameter Values:{row["PARAMETER"]}'
            },
            {
                "role": "assistant",
                "content": str(json_content)
            }
        ]
    }
    # Append the message object to the combined data list
    combined_data.append(message_obj)

# Save the combined data in a JSONL file
with open('../Openai/dataset.jsonl', 'w') as outfile:
    for entry in combined_data:
        json.dump(entry, outfile)
        outfile.write('\n')
