

import pandas as pd
import os
import json

def process_json_files_with_labels(base_dir, combined_df):
    all_data = []

    for _, row in combined_df.iterrows():
        file_name = row['File Name']
        file_path = os.path.join(base_dir, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

                # Initialize file_data with additional columns from the DataFrame
                file_data = {
                    'File Name': file_name,
                    'Side': row['side'],
                    'Type': row['Type'],
                    'Primary Label': row['Primary label']
                }

                # Process the JSON data, but stop after the 76th second
                for second, payload in enumerate(data.get('payloads', []), start=1):
                    if second > 76:  # Stop adding data after the 76th second
                        break
                    elements = payload.split(';')
                    for ms, element in enumerate(elements, start=1):
                        xyz_values = element.split(',')[-3:]
                        if len(xyz_values) == 3:
                            # Creating unique column names for each millisecond
                            col_name_x = f"S{second} MS{ms} X"
                            col_name_y = f"S{second} MS{ms} Y"
                            col_name_z = f"S{second} MS{ms} Z"
                            file_data[col_name_x] = xyz_values[0]
                            file_data[col_name_y] = xyz_values[1]
                            file_data[col_name_z] = xyz_values[2]

                    # Add an additional column at the end of each second with a placeholder value
                    # Assuming the value is to be directly obtained or is a constant for demonstration
                    # This can be replaced or calculated as per actual requirements
                    file_data[f"S{second} Speed Placeholder"] = "Placeholder Value"

                all_data.append(file_data)
        else:
            print(f"{file_name}: File not found.")

    df = pd.DataFrame(all_data)
    return df


# Loading the Excel file
excel_path = '20230509 labeling.xlsx'  

# Reading the sheets
accidents_df = pd.read_excel(excel_path, sheet_name='תאונות')
no_accidents_df = pd.read_excel(excel_path, sheet_name='לא-תאונות')



# Combining both DataFrames
combined_df = pd.concat([
    accidents_df[['File Name', 'side', 'Type', 'Primary label']],
    no_accidents_df[['File Name', 'side', 'Type', 'Primary label']]
])

# Set the correct base directory where JSON files are stored
base_dir = r'C:\Users\idan_sa\PycharmProjects\accidents and not accidents'
df_processed = process_json_files_with_labels(base_dir, combined_df)

# Save the processed data to a CSV file or use as needed
csv_path = os.path.join(r'C:\Users\idan_sa\PycharmProjects\pythonProject\New folder', 'processed_data_with_labels_1.csv')
df_processed.to_csv(csv_path, index=False)

print(f"Processed data saved to: {csv_path}")
