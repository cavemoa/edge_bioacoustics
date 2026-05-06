from pathlib import Path

import pandas as pd

def get_single_word_labels(csv_file_path, column_name='label'):
    """
    Extracts single-word labels from a specified CSV file.
    
    Args:
        csv_file_path (str): The path to the perch label.csv file.
        column_name (str): The name of the column containing the labels.
                           Defaults to 'label'.
                           
    Returns:
        list: A list of single-word labels.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Fallback to the first column if the specified column name isn't found
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found. Using the first column '{df.columns[0]}' instead.")
            column_name = df.columns[0]
            
        # Drop any empty/NaN rows in the label column and convert to string
        labels = df[column_name].dropna().astype(str).tolist()
        
        # Filter for labels that only contain one word (no spaces)
        single_word_labels = [label for label in labels if len(label.split()) == 1]
        
        return single_word_labels
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Please check the path.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    file_path = Path(__file__).with_name('perch_label.csv')
    
    # Run the extraction
    extracted_labels = get_single_word_labels(file_path, column_name='label')
    
    # Display the results
    if extracted_labels:
        print(f"Successfully extracted {len(extracted_labels)} single-word labels:")
        print("-" * 40)
        for word in extracted_labels:
            print(word)
