import os
import pandas as pd

# Directory containing the Excel sheets
directory = (
    "/home/salnahari/testing_dir/tt-metal/ttnn/ttnn/experimental_tracer/models"  # Replace with your directory path
)

# List to store dataframes
dataframes = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):  # Check for Excel files
        file_path = os.path.join(directory, filename)

        # Read the Excel file
        df = pd.read_excel(file_path)

        # Add a new column with the sheet name (file name without extension)
        df["SheetName"] = os.path.splitext(filename)[0]

        # Append the dataframe to the list
        dataframes.append(df)

# Combine all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new Excel file
output_file = "/home/salnahari/testing_dir/tt-metal/ttnn/ttnn/experimental_tracer/combined_excel.xlsx"  # Replace with your desired output path
combined_df.to_excel(output_file, index=False)

print(f"Combined Excel file saved to: {output_file}")
