import pandas as pd
import openai
import os


# Initialize OpenAI
openai.api_key = "openai.api_key"  # replace with your API key

# Load the Excel file
file_path = '../Openai/DATA.xlsx'
df = pd.read_excel(file_path)

all_results = []
results_data = []

for idx, row in df.iterrows():
    LOGS = row['LOGS']
    PARAMETER = row['PARAMETER']
    #print(f"Processing row {idx + 1} with LOGS: '{LOGS}' and PARAMETER: '{PARAMETER}'")

    try:

        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:wholescale::8KwvPB6L",
            messages=[
                {"role": "system",
                 "content": """Analyze the provided system log sequence and parameter values for potential anomalies using a structured approach. Present your findings in a JSON format with the key 'anomaly' indicating 'yes' or 'no'. Focus on key categories and their associated keywords.
                    Instructions:
                    1. Hierarchical Transformer Structure: Implement a hierarchical transformer structure for deep analysis of system logs. This structure should effectively capture the semantic nuances in both log template sequences and parameter values, which are essential for accurate anomaly detection.
                    2. Log Sequence and Parameter Value Encoding: 
                       - Log Sequence Encoder: Transform log template sequences into fixed-dimension vectors. This process should encapsulate the semantic content of log templates and the contextual relevance of the log sequence.
                       - Parameter Value Encoder: Convert parameter values into fixed-dimension vectors, emphasizing the semantic significance of parameters to identify different types of performance anomalies.
                    3. Attention-Based Classification: Utilize an attention mechanism in your classification model. This mechanism should dynamically assign varying weights to the representations of log sequences and parameter sequences, prioritizing the aspect more indicative of an anomaly.
                    4. Robustness and Effectiveness: Ensure that your analysis is both robust and effective. Your approach should be capable of managing unstable log data and demonstrate superior performance compared to other log-based anomaly detection methods, similar to the achievements of HitAnomaly.
                    5. Future Directions: Explore the possibility of integrating the transformer structure into a log-based anomaly prediction task. The goal is to predict potential anomalies before they manifest.
                    """},
                {"role": "user",
                 "content": "Here is the Log_Sequence:" + str(LOGS) + "Here is the Parameter_Values:" + str(PARAMETER)}
            ]
        )
        json_output_from_code = eval(completion.choices[0].message['content'])
        # Assuming 'Json' column is a string representation of a dictionary
        json_output_from_excel = eval(row['Json'])
        result = json_output_from_code == json_output_from_excel
        print(result)
        #print(json_output_from_code)
        #print(json_output_from_excel)
        #print("Comparison successful." if result else "Comparison failed.")
        all_results.append(result)

    except Exception as e:
        #print(f"Error processing row with LOGS: {LOGS} and PARAMETER: {PARAMETER}. Error: {e}")
        result = False
        json_output_from_code = "ERROR"

    # Append results to the list
    results_data.append({
        'LOGS': LOGS,
        'PARAMETER': PARAMETER,
        'Json from Excel': row['Json'],
        'Json from Code': json_output_from_code,
        'Result': result
    })

# Create a DataFrame from the results data
results_sheet = pd.DataFrame(results_data)

# Save the results_sheet to a new Excel sheet
results_sheet.to_excel('Test_Results.xlsx', index=False)

# Calculate the cumulative accuracy
accuracy = sum(all_results) / len(all_results) * 100
print(f"Cumulative Accuracy: {accuracy:.2f}%")
