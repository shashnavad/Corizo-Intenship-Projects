import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data from a CSV file
data = pd.read_csv('Data/Cleaned/New/completeCleanedLabeled.csv')

# Randomize data order
data = data.sample(frac=1).reset_index(drop=True)

# Split the data into training and testing sets
train_set, test_set = train_test_split(data, test_size=0.2, random_state=7)

# Save the training and testing sets into separate CSV files
train_set.to_csv('Data/Cleaned/New/trainCleanedLabeled2.csv', index=False)
test_set.to_csv('Data/Cleaned/New/testCleanedLabeled2.csv', index=False)
