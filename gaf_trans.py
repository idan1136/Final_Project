import matplotlib.pyplot as plt
import pandas as pd
from pyts.image import GramianAngularField
import numpy as np
from os import path, makedirs
from tqdm import tqdm
from joblib import Parallel, delayed

# Directory where the images will be saved
output_dir = path.join('45-76_gaf_images')

# Creating the required directories
for folder in ['ACCIDENT', 'NOT ACCIDENT']:
    for axis_folder in ['X', 'Y', 'Z']:
        makedirs(path.join(output_dir, folder, axis_folder), exist_ok=True)

# Load the dataset from the CSV file
df = pd.read_csv('processed_data_with_labels.csv')

# Function to generate and save images for each event in the dataset
def process_row(index, row):
    primary_label = row['Primary Label']
    label_folder = 'ACCIDENT' if primary_label == 1 else 'NOT ACCIDENT'

    # Loop through each axis
    for axis in ['X', 'Y', 'Z']:
        axis_data = []

        # Loop through all 76 seconds, collecting 50 ms samples for each second
        for second in range(45, 76):  # S45 to S76
            for ms in range(1, 51):  # MS1 to MS50 for each second
                col_name = f'S{second} MS{ms} {axis}'
                axis_data.append(row[col_name])

        # Convert list to numpy array
        axis_data = np.array(axis_data)

        # Ensure we have sufficient data before proceeding
        if len(axis_data) < 50 * (76 - 45):  # 50 samples per second, 31 seconds
           continue

        # Reshape the data for the GAF transformation
        data = axis_data.reshape(1, -1)

        # Transform the time series into an image using Gramian Angular Field (GAF)
        gaf = GramianAngularField(method='summation')
        image = gaf.fit_transform(data)[0]

        # Save the image
        image_filename = f"{index + 1}_{primary_label}_{axis}.png"
        image_filepath = path.join(output_dir, label_folder, axis, image_filename)
        plt.imsave(image_filepath, image, cmap="jet")

# Generate and save images based on the dataset using parallel processing
Parallel(n_jobs=-1)(delayed(process_row)(index, row) for index, row in tqdm(df.iterrows(), total=len(df)))

print('Process completed successfully')
