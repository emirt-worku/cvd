
import seaborn as sns
import matplotlib.pyplot as plt
# Extract feature names
feature_names = df.columns.tolist()

# Print the feature names
print("Feature Names:", feature_names)

from sklearn.model_selection import train_test_split, cross_val_score,cross_validate
x = df.drop('HadHeartAttack',axis = 1)
y = df['HadHeartAttack']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=.2,stratify=y)
# Let us first distingush datasets into num and object
num = []
cat = []
for i,col in enumerate(df.columns):
    if df[col].nunique() > 5:
        num.append(col)
    else:
        cat.append(col)
        df[col] = df[col].astype('object')


# Generate histograms for numeric columns
df.select_dtypes(['float', 'int']).hist(bins=50, figsize=(15, 10))

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file
plt.savefig('histogram_output 2.png')  # You can change the filename and format as needed

# Display the plot
plt.show()


# Set the figure size for better readability
plt.figure(figsize=(13, 6))

# Create boxplots for 'BMI' and 'SleepHours'
sns.boxplot(num_f)

# Set the title of the plot
plt.title('Boxplots of the numerical features')
plt.savefig('Boxplots  ') 
# Display the plot
plt.show()


# Loop through each numerical column in the dataframe num_f
for column in num_f.columns:
    # Calculate Z-scores for the column
    z = np.abs((num_f[column] - num_f[column].mean()) / num_f[column].std())
    
    # Set the threshold for outliers (Z-score > 3)
    outliers = num_f[column][z > 3]
    
    # Print the outliers for the current column
    if not outliers.empty:
        print(f"Outliers detected in {column}:")
        print(outliers)
        print("\n")
    else:
        print(f"No outliers detected in {column}.")
        print("\n")
# Function to cap outliers beyond 3 standard deviations
def cap_outliers(df, column):
    mean = df[column].mean()
    std_dev = df[column].std()
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    df[column] = np.clip(df[column], lower_bound, upper_bound)
df_cleaned = num_f.copy()

# BMI - Replace '0' values with median (if there are any '0' values)
BMI_median = df_cleaned['BMI'][df_cleaned['BMI'] != 0].median()
df_cleaned['BMI'] = df_cleaned['BMI'].replace(0, BMI_median)

# Sleephours - Replace '0' values with median (if applicable)
SleepHours_median = df_cleaned['SleepHours'][df_cleaned['SleepHours'] != 0].median()
df_cleaned['SleepHours'] = df_cleaned['SleepHours'].replace(0,SleepHours_median)

# Print the summary to verify the replacement
print(f"New BMI median after imputation: {BMI_median}")
print(f"New Sleephours median after imputation: {SleepHours_median}")
import pandas as pd

# Assuming 'df_cleaned' is your cleaned DataFrame
data.to_csv('cleaned_data.csv', index=False)
