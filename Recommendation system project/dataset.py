import pandas as pd

# Load the datasets with a different encoding
train_df = pd.read_csv('train.csv', encoding='ISO-8859-1')  # Adjust the file path if needed
product_desc_df = pd.read_csv('product_descriptions.csv', encoding='ISO-8859-1')  # Adjust the file path if needed

# Merge the datasets on the 'product_uid' column
combined_df = pd.merge(train_df, product_desc_df, on='product_uid', how='inner')

# Show the first few rows of the combined dataframe
print(combined_df.head())

# Optionally, save the combined dataframe to a new CSV file
combined_df.to_csv('combined_dataset.csv', index=False)
