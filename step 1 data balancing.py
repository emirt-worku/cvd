import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

# Assuming the dataset is already loaded into 'df' and target column is 'target'

# Filter only class 0
class_0 = df[df['HadHeartAttack'] == 0]

# Calculate the size of each sample
sample_size = len(class_0) // 17

# Split class 0 into 17 subsets
samples = []
for i in range(17):
    start_idx = i * sample_size
    end_idx = (i + 1) * sample_size if i < 16 else len(class_0)  # Include remaining rows in the last sample
    samples.append(class_0.iloc[start_idx:end_idx])

# Save each subset as a separate file (optional)
for idx, sample in enumerate(samples):
    filename = f'class_0_sample_{idx + 1}.csv'
    sample.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(sample)} rows.")

# Verify the sizes of each subset
for idx, sample in enumerate(samples):
    print(f"Sample {idx + 1}: {len(sample)} rows")

# Assuming the dataset is already loaded into 'df' and the target column is 'HadHeartAttack'
target_col = "HadHeartAttack"  # Replace with the correct target column name

# Separate class 0 and class 1
class_0 = df[df[target_col] == 0]
class_1 = df[df[target_col] == 1]

# Calculate the size of each sample
sample_size = len(class_0) // 17

# Split class 0 into 17 subsets
samples = []
for i in range(17):
    start_idx = i * sample_size
    end_idx = (i + 1) * sample_size if i < 16 else len(class_0)
    samples.append(class_0.iloc[start_idx:end_idx])

# Initialize results list to store evaluation metrics
results = []

# Loop through each class 0 sample
for idx, sample in enumerate(samples):
    print(f"Processing sample {idx + 1}...")
    
    # Merge class 1 with the current class 0 sample
    merged_data = pd.concat([sample, class_1])
    
    # Shuffle the merged dataset
    merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the merged dataset to a CSV file
    merged_filename = f'merged_dataset_sample_{idx + 1}.csv'
    merged_data.to_csv(merged_filename, index=False)
    print(f"Saved merged dataset: {merged_filename}")
    
    # Split features and target
    x = merged_data.drop(columns=[target_col])
    y = merged_data[target_col]
    
    # Encode categorical variables using Label Encoding
    for col in x.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
    
    # Ensure all transformations maintain the same length
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Split into train and test sets (e.g., 80/20 split)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Train a simple model (e.g., Logistic Regression)
    model = LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter for convergence
    model.fit(x_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(x_test)
    
    # Store the metrics
    metrics = {
        'sample': idx + 1,
        'accuracy': round(accuracy_score(y_test, y_pred), 2),
        'precision': round(precision_score(y_test, y_pred), 2),
        'recall': round(recall_score(y_test, y_pred), 2),
        'f1_score': round(f1_score(y_test, y_pred), 2)
    }
    results.append(metrics)
    print(f"Sample {idx + 1}: Metrics {metrics}")

# Save results to a CSV file
results_df = pd.DataFrame(results)

# Ensure the table is formatted to two decimal places for display
results_df = results_df.round(2)

# Save the formatted results to a CSV file
results_df.to_csv('model_results.csv', index=False)

print("Saved model evaluation results to 'model_results.csv'.")
