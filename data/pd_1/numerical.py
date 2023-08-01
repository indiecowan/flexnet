import pandas as pd

def convert_to_numerical(old_csv_path, new_csv_path):
    # Load the dataset
    df = pd.read_csv(old_csv_path)
    
    # Drop the Patient Id column
    df = df.drop('Patient Id', axis=1)
    
    # Convert the Level column to numerical
    # First, let's check what unique values are in the 'Level' column
    print("Unique values in 'Level' column before conversion:", df['Level'].unique())
    
    # You can then map these to numerical values. For example,
    df['Level'] = df['Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    # Verify that the 'Level' column has been converted correctly
    print("Unique values in 'Level' column after conversion:", df['Level'].unique())
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(new_csv_path, index=False)

# Use the function
old_csv_path = 'og_patient_data.csv'
new_csv_path = 'patient_data.csv.csv'
convert_to_numerical(old_csv_path, new_csv_path)
