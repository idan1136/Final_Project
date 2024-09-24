import os
import json
import pandas as pd
import math
from math import dist
import numpy as np
from datetime import datetime, timedelta
from statistics import mean, stdev, variance, median
import ctypes
import argparse


def json_to_csv_conv(json_directory):
    """ Take a folder with json files, convert the json into csv and save it """
    columns_csv = ['Current Second', 'Time Stamp', 'Latitude',
                   'Longitude', 'Azimute', 'Speed',
                   'x_1', 'y_1', 'z_1',
                   'x_2', 'y_2', 'z_2',
                   'x_3', 'y_3', 'z_3',
                   'x_4', 'y_4', 'z_4',
                   'x_5', 'y_5', 'z_5',
                   'x_6', 'y_6', 'z_6',
                   'x_7', 'y_7', 'z_7',
                   'x_8', 'y_8', 'z_8',
                   'x_9', 'y_9', 'z_9',
                   'x_10', 'y_10', 'z_10',
                   'x_11', 'y_11', 'z_11',
                   'x_12', 'y_12', 'z_12',
                   'x_13', 'y_13', 'z_13',
                   'x_14', 'y_14', 'z_14',
                   'x_15', 'y_15', 'z_15',
                   'x_16', 'y_16', 'z_16',
                   'x_17', 'y_17', 'z_17',
                   'x_18', 'y_18', 'z_18',
                   'x_19', 'y_19', 'z_19',
                   'x_20', 'y_20', 'z_20',
                   'x_21', 'y_21', 'z_21',
                   'x_22', 'y_22', 'z_22',
                   'x_23', 'y_23', 'z_23',
                   'x_24', 'y_24', 'z_24',
                   'x_25', 'y_25', 'z_25',
                   'x_26', 'y_26', 'z_26',
                   'x_27', 'y_27', 'z_27',
                   'x_28', 'y_28', 'z_28',
                   'x_29', 'y_29', 'z_29',
                   'x_30', 'y_30', 'z_30',
                   'x_31', 'y_31', 'z_31',
                   'x_32', 'y_32', 'z_32',
                   'x_33', 'y_33', 'z_33',
                   'x_34', 'y_34', 'z_34',
                   'x_35', 'y_35', 'z_35',
                   'x_36', 'y_36', 'z_36',
                   'x_37', 'y_37', 'z_37',
                   'x_38', 'y_38', 'z_38',
                   'x_39', 'y_39', 'z_39',
                   'x_40', 'y_40', 'z_40',
                   'x_41', 'y_41', 'z_41',
                   'x_42', 'y_42', 'z_42',
                   'x_43', 'y_43', 'z_43',
                   'x_44', 'y_44', 'z_44',
                   'x_45', 'y_45', 'z_45',
                   'x_46', 'y_46', 'z_46',
                   'x_47', 'y_47', 'z_47',
                   'x_48', 'y_48', 'z_48',
                   'x_49', 'y_49', 'z_49',
                   'x_50', 'y_50', 'z_50']

    err_fls = 0

    # Iterate over folders in directory
    for folder_name in os.listdir(json_directory):
        # Get file path
        folder_path = os.path.join(json_directory, folder_name)
        # Iterate over files in folder
        for filename in os.listdir(folder_path):

            # Get file path
            file_path = os.path.join(folder_path, filename)

            # Opening JSON file
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Define Results List
            results_list = []

            # Iterate over each second within the payload
            for current_second in data['payloads']:
                # Replace any Semi Colons with comma
                clean_record = current_second.replace(';', ',')

                # Convert String to List - Split by Comma
                list_record = clean_record.split(',')

                # Delete first 5 values - Irrelevant Data
                del list_record[0:5]

                # Append to Results List
                results_list.append(list_record)

            try:
                # Create Data frame - Pass in Results List & Define Column Names
                results_df = pd.DataFrame(data=results_list, columns=columns_csv)

                # Get file name and path without extension
                output_path = file_path.replace('json', 'csv')

                # Save results to CSV
                results_df.to_csv(output_path, index=False)

            except:
                print(filename)
                err_fls += 1
                continue

    print(f'Number of files that been damaged and not saved in csv file folder: {err_fls}')

def delete_unwanted_data(directory):
    """ This function get directory and delete file from json and csv folder """
    if os.path.exists(directory):
        os.remove(directory)
        print(f'{directory} has been deleted')

    csv_path = directory.replace('json', 'csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f'{csv_path} has been deleted\n')

    print('Done')



def create_143_table_from_json(json_dir, out_filename):
    """Get a directory of JSON folder files and create a CSV table with 143 content."""

    # Create columns
    columns = ['filename', 'date', 'pocsag', 'IP', 'speed', 'azimuth', 'latitude', 'longitude', 'x_143', 'y_143',
               'z_143', 'accident_direction', 'label']

    # Create dictionary that will be the body of the table
    body = {col: [] for col in columns}

    # Iterate over each folder in the directory
    for json_folder in os.listdir(json_dir):

        # Path for specific JSON folder (accidents / not_accidents)
        json_folder_path = os.path.join(json_dir, json_folder)

        # Iterate over files in the directory
        for filename in os.listdir(json_folder_path):
            # Check if the file is a JSON file
            if not filename.endswith('.json'):
                print(f"Skipping non-JSON file: {filename}")
                continue

            # Path to the actual JSON file
            json_file_path = os.path.join(json_folder_path, filename)

            # Read JSON file
            with open(json_file_path, encoding="utf-8") as f:
                json_data = json.load(f)

            first_msg_speed = float(json_data['speed'])

            # Check if we encounter a quality case of 253
            if first_msg_speed < 0 or first_msg_speed > 400:
                delete_unwanted_data(json_file_path)
                continue

            body['speed'].append(first_msg_speed)

            # Take full 143 message and split it by comma
            lim_143 = json_data['lim_143'].split(',')

            # Add axis to dictionary
            body['x_143'].append(float(lim_143[1]) / 55)
            body['y_143'].append(float(lim_143[2]) / 55)
            body['z_143'].append(float(lim_143[3]) / 55)

            # Add the accident direction (number between 1 to 9)
            body['accident_direction'].append(int(lim_143[4]))

            # Additional data from the JSON
            body['filename'].append(filename)
            body['latitude'].append(json_data['lat'])
            body['longitude'].append(json_data['lon'])
            body['date'].append(json_data['date'])
            body['azimuth'].append(json_data['head'])
            body['pocsag'].append(json_data['pocsag'])
            body['IP'].append(json_data['IP'])

            if json_folder.endswith('not_accidents') or json_folder.endswith('false'):
                body['label'].append(0)
            else:
                body['label'].append(1)

    # Create DataFrame - Pass in Results List & Define Column Names
    results_df = pd.DataFrame.from_dict(body)

    # Save results to CSV
    table_folder_path = json_dir.replace('json_files', '143_table')
    table_folder_path = os.path.join(table_folder_path, out_filename)
    results_df.to_csv(f'{table_folder_path}.csv', index=False)
    print(f"CSV saved at: {table_folder_path}.csv")
    print('Done')


def accident_direction_mapping(num):
    """ Map the accident direction number to string representation

    :param num: numeric representation of the accident's accident_direction
    :type num: int
    """

    if num in (1, 2, 3):
        return 'front'
    elif num in (4, 5, 6):
        return 'back'
    elif num == 7:
        return 'left'
    elif num == 8:
        return 'right'
    elif num == 9:
        return 'roll'
    else:
        return None


def local_spike_counter_from_window(window_axi, threshold=0.05):
    """ that function get the x axi for specific second and return the number of spikes

    :param threshold: parameter that determine the size of the spike (G-force)
    :type threshold: float

    :param window_axi: list with x parameter from current second (out of 76)
    :type window_axi: list<float>

    """
    up = 1
    down = 0
    nothing = -1
    spike_counter = 0
    accumulate_power = 0
    trend = nothing
    last = window_axi[0]
    details = []
    for i in range(1, len(window_axi)):
        if window_axi[i] >= last and trend in (up, nothing):
            trend = up
        elif window_axi[i] < last and trend in (down, nothing):
            trend = down
        elif window_axi[i] < last and trend == up:
            spike_counter += 1
            trend = down
            details.append((i - 1, window_axi[i - 1], spike_counter))
        elif window_axi[i] >= last and trend == down:
            spike_counter += 1
            trend = up
            details.append((i - 1, window_axi[i - 1], spike_counter))

        last = window_axi[i]
        accumulate_power += abs(i)

    # after that we find local min and max points
    res = 0
    for i in range(1, len(details)):
        if abs(details[i][1] - details[i - 1][1]) >= threshold:
            res += 1

    return res


def spike_counter_in_current_second(axi_x):
    """ that function get the x axi for specific second and return the number of spikes

    :param axi_x: list with x parameter from current second (out of 76)
    :type axi_x: list<float>
    """
    spike_counter = 0

    for i in range(0, (len(axi_x) - 3)):
        spike0, spike1, spike2 = float(axi_x[i]), float(axi_x[i + 1]), float(axi_x[i + 2])
        if ((spike0 < spike1 > spike2) or (spike0 > spike1 < spike2)) and (abs(spike0 - spike1) > 0.05) and (
                abs(spike1 - spike2) > 0.05):
            spike_counter += 1

    return spike_counter


def features_calculator(second_acc, second_range_and_type):
    """ calculator the features and return it as dictionary

    :param second_acc: the axi parameters from the selected range - for example: from second 45 to second 55 from the x axi
    :type second_acc: list<float>

    :param second_range_and_type: the range of second and type of axi - for example: 45_55_x
    :type second_range_and_type: string

    :return features_dict: dictionary that contain all the features_dict
    :type: dictionary

    Explain features:
    - Maximum values of the axis between the given range
    - Minimum values of the axis between the given range
    - Maximum absolute value of the axis between maximum and the minimum values
    - Distance between the maximum and the minimum of the  axis between the given range
    - Mean value of the axis between the given range
    - Median value of the axis between the given range
    - Variance value of the axis between the given range
    - Standard deviation value of the axis between the given range
    - The sum of the values of the axis between the given range
    """

    # dictionary that will contain all the features
    features_dict = {}

    max_f = max(second_acc)
    min_f = min(second_acc)

    features_dict[f'max_{second_range_and_type}'] = max_f
    features_dict[f'min_{second_range_and_type}'] = min_f
    features_dict[f'max_abs_{second_range_and_type}'] = max(abs(min_f), abs(max_f))
    features_dict[f'dis_max_min_{second_range_and_type}'] = dist([min_f], [max_f])
    features_dict[f'mean_{second_range_and_type}'] = mean(second_acc)
    features_dict[f'median_{second_range_and_type}'] = median(second_acc)
    features_dict[f'variance_{second_range_and_type}'] = variance(second_acc)
    features_dict[f'std_{second_range_and_type}'] = stdev(second_acc)
    features_dict[f'sum_{second_range_and_type}'] = sum(second_acc)

    return features_dict


def dataset_creator(csv_dir, out_filename=None, table_143_filename=None):
    """ Function that get csv directory and out file name and return csv dataset

    :param csv_dir: path to the folder contains the csv files
    :type csv_dir: String

    :param out_filename:  output csv file name
    :type out_filename: String

    :param table_143_filename: file name for the 143 table (csv file that contain all the first messages data)
    :type table_143_filename: String

    """

    # path to the 143 table that contain all the 143 messages
    table_file_path = f'143_csv/{table_143_filename}.csv'

    # read 143 table as DataFrame
    df_143 = pd.read_csv(table_file_path)

    # take out the filename column from the 143 table
    filenames_143 = df_143['filename'].tolist()

    # create columns
    columns = ['filename',
               'date',
               'pocsag',
               'IP',
               'speed_143',
               'azimuth_143',
               'latitude_143',
               'longitude_143',
               'x_143',
               'y_143',
               'z_143',
               'accident_direction',
               'accident_four_direction',
               'max_45_55_x',
               'max_45_55_y',
               'max_45_55_z',
               'min_45_55_x',
               'min_45_55_y',
               'min_45_55_z',
               'max_abs_45_55_x',
               'max_abs_45_55_y',
               'max_abs_45_55_z',
               'max_48_52_x',
               'max_48_52_y',
               'max_48_52_z',
               'min_48_52_x',
               'min_48_52_y',
               'min_48_52_z',
               'max_abs_48_52_x',
               'max_abs_48_52_y',
               'max_abs_48_52_z',
               'is_grater08_abs_x',
               'is_grater08_abs_y',
               'is_grater18_abs_z',
               'mean_45_55_x',
               'mean_45_55_y',
               'mean_45_55_z',
               'total_accelerated_45_55_xyz',
               'median_45_55_x',
               'median_45_55_y',
               'median_45_55_z',
               'std_45_55_x',
               'std_45_55_y',
               'std_45_55_z',
               'variance_45_55_x',
               'variance_45_55_y',
               'variance_45_55_z',
               'mean_48_52_x',
               'mean_48_52_y',
               'mean_48_52_z',
               'total_accelerated_48_52_xyz',
               'median_48_52_x',
               'median_48_52_y',
               'median_48_52_z',
               'std_48_52_x',
               'std_48_52_y',
               'std_48_52_z',
               'variance_48_52_x',
               'variance_48_52_y',
               'variance_48_52_z',
               'over_15_on_55',
               'over_15_on_70',
               'sum_45_55_x',
               'sum_45_55_y',
               'sum_45_55_z',
               'sum_48_52_x',
               'sum_48_52_y',
               'sum_48_52_z',
               'dist_max_min_45_55_x',
               'dist_max_min_45_55_y',
               'dist_max_min_45_55_z',
               'dist_max_min_48_52_x',
               'dist_max_min_48_52_y',
               'dist_max_min_48_52_z',
               'th_05_spikes_55_76_x',
               'th_10_spikes_55_76_x',
               'th_05_spikes_55_76_y',
               'th_10_spikes_55_76_y',
               'th_05_spikes_55_76_z',
               'th_10_spikes_55_76_z',
               'th_15_spikes_55_76_z',
               'th_25_spikes_55_76_z',
               'th_05_more_then_5_spikes_55_76_x',
               'th_05_more_then_10_spikes_55_76_x',
               'th_05_more_then_5_spikes_55_76_z',
               'th_05_more_then_05_x_and_th_25_more_then_05_z_spikes_55_76',
               'th_05_more_then_10_x_and_th_25_more_then_05_z_spikes_55_76',
               'x_above_th_in_millisec',
               'y_above_th_in_millisec',
               'z_above_th_in_millisec',
               'spikes_55_76',
               'label']

    # create dictionary that will be the body of the table
    body = {col: [] for col in columns}

    # every inner folder in the json_folder file
    for csv_folder in os.listdir(csv_dir):

        # path for specific json folder (accidents / not_accidents)
        csv_folder_path = os.path.join(csv_dir, csv_folder)

        # Iterate over files in directory

        for filename in os.listdir(csv_folder_path):
            if filename.endswith('.csv'):
                # path to the actual json file
                csv_file_path = os.path.join(csv_folder_path, filename)

                # check if the file is in the table
                json_rep = filename.replace('csv', 'json')
                if not (json_rep in filenames_143):
                    print(f'Error: {json_rep} is not in 143 table')
                    continue

                # extract row in 143 related to specific accident (This row will contain all the first message data)
                accident_row_in_143 = df_143.loc[df_143['filename'] == json_rep]

                # extract direction by number
                accident_direction_num = accident_row_in_143['accident_direction'].values[0]

                # read the csv accident file as dataframe
                df_acc = pd.read_csv(csv_file_path)

                # Get max Speed on the 55 second and after the 70's sec then it will return 1 if greater then 15 else 0
                max_speed_55 = 1 if max(df_acc.iloc[54:55, 5]) > 15 else 0
                max_speed_after_70 = 1 if max(df_acc.iloc[-6:, 5]) > 15 else 0

                # Get axis columns
                df_axis = df_acc.iloc[:, 6:]

                # Create Accelerometer list
                second_acc_45_55_x, second_acc_45_55_y, second_acc_45_55_z = [], [], []
                second_acc_48_52_x, second_acc_48_52_y, second_acc_48_52_z = [], [], []

                # create lists for spikes windows
                window_55_76_x, window_55_76_y, window_55_76_z = [], [], []
                window_full_x, window_full_y, window_full_z = [], [], []

                # Iterator that run over all the seconds (Rows)
                for sec_out_of_76, row in df_axis.iterrows():

                    # Create Accelerometer lists for current second
                    axi_x, axi_y, axi_z = [], [], []

                    # Run over all the columns of the axis
                    for i in range(0, 150, 3):
                        axi_x.append(row[i])
                        axi_y.append(row[i + 1])
                        axi_z.append(row[i + 2])

                    # all 76 seconds by axi
                    window_full_x.extend(axi_x)
                    window_full_y.extend(axi_y)
                    window_full_z.extend(axi_z)

                    # If current second is between the 55's and the 76's second check for spikes
                    if 54 < sec_out_of_76:
                        window_55_76_x.extend(axi_x)
                        window_55_76_y.extend(axi_y)
                        window_55_76_z.extend(axi_z)

                    # If current second is between the 45's and the 55's second
                    if 44 < sec_out_of_76 < 56:
                        # Append axis by seconds
                        second_acc_45_55_x.extend(axi_x)
                        second_acc_45_55_y.extend(axi_y)
                        second_acc_45_55_z.extend(axi_z)

                        # If current second is between the 48's and the 52's second
                        if 47 < sec_out_of_76 < 53:
                            second_acc_48_52_x.extend(axi_x)
                            second_acc_48_52_y.extend(axi_y)
                            second_acc_48_52_z.extend(axi_z)

                # get the number of spikes from window
                th_05_spikes_55_76_x = local_spike_counter_from_window(window_55_76_x, threshold=0.05)
                th_10_spikes_55_76_x = local_spike_counter_from_window(window_55_76_x, threshold=0.10)
                th_05_spikes_55_76_y = local_spike_counter_from_window(window_55_76_y, threshold=0.05)
                th_10_spikes_55_76_y = local_spike_counter_from_window(window_55_76_y, threshold=0.10)
                th_05_spikes_55_76_z = local_spike_counter_from_window(window_55_76_z, threshold=0.05)
                th_10_spikes_55_76_z = local_spike_counter_from_window(window_55_76_z, threshold=0.10)
                th_15_spikes_55_76_z = local_spike_counter_from_window(window_55_76_z, threshold=0.15)
                th_25_spikes_55_76_z = local_spike_counter_from_window(window_55_76_z, threshold=0.25)

                # calculate features for 45-55 total_seconds
                features_45_55_x = features_calculator(second_acc_45_55_x, '45_55_x')
                features_45_55_y = features_calculator(second_acc_45_55_y, '45_55_y')
                features_45_55_z = features_calculator(second_acc_45_55_z, '45_55_z')

                # calculate features for 48-52 total_seconds
                features_48_52_x = features_calculator(second_acc_48_52_x, '48_52_x')
                features_48_52_y = features_calculator(second_acc_48_52_y, '48_52_y')
                features_48_52_z = features_calculator(second_acc_48_52_z, '48_52_z')

                # check if the axis has bypass a certain threshold (0.8 fo x and y and 1.8 for z)
                is_grater08_abs_x = 1 if features_48_52_x['max_abs_48_52_x'] > 0.8 else 0
                is_grater08_abs_y = 1 if features_48_52_y['max_abs_48_52_y'] > 0.8 else 0
                is_grater18_abs_z = 1 if features_48_52_z['max_abs_48_52_z'] > 1.8 else 0

                # check how much x's (or any other axi) we get every single second
                xyz_per_sec = len(window_full_x) / 76

                # now we want to get the times x (or any other axi) show in
                sample_every_millisec = 1000 / xyz_per_sec

                # counters for the time (Milli seconds) each axi go above some threshold
                x_above_th_in_millisec, y_above_th_in_millisec, z_above_th_in_millisec = 0, 0, 0

                # iterate over all the values of the axis and count how much time the value was over the threshold
                for i in range(len(window_full_x)):
                    if abs(window_full_x[i]) > 1.2: x_above_th_in_millisec += 1
                    if abs(window_full_y[i]) > 0.9: y_above_th_in_millisec += 1
                    if abs(window_full_z[i]) > 2.0: z_above_th_in_millisec += 1

                # update feature with the number of Milli seconds that the values of specific axi was over tge threshold
                body['x_above_th_in_millisec'].append(x_above_th_in_millisec * sample_every_millisec)
                body['y_above_th_in_millisec'].append(y_above_th_in_millisec * sample_every_millisec)
                body['z_above_th_in_millisec'].append(z_above_th_in_millisec * sample_every_millisec)

                # 143 (fist massage) data and speed after 55/76
                body['filename'].append(accident_row_in_143['filename'].values[0])
                body['date'].append(accident_row_in_143['date'].values[0])
                body['pocsag'].append(accident_row_in_143['pocsag'].values[0])
                body['IP'].append(accident_row_in_143['IP'].values[0])
                body['speed_143'].append(accident_row_in_143['speed'].values[0])
                body['azimuth_143'].append(accident_row_in_143['azimuth'].values[0])
                body['latitude_143'].append(accident_row_in_143['latitude'].values[0])
                body['longitude_143'].append(accident_row_in_143['longitude'].values[0])
                body['x_143'].append(accident_row_in_143['x_143'].values[0])
                body['y_143'].append(accident_row_in_143['y_143'].values[0])
                body['z_143'].append(accident_row_in_143['z_143'].values[0])
                body['accident_direction'].append(accident_direction_num)
                body['over_15_on_55'].append(max_speed_55)
                body['over_15_on_70'].append(max_speed_after_70)
                body['label'].append(accident_row_in_143['label'].values[0])

                # use mapping function that map the direction of the accidents from a number into string representation
                body['accident_four_direction'].append(accident_direction_mapping(accident_direction_num))

                # general statistics for window 45 - 55
                body['max_45_55_x'].append(features_45_55_x['max_45_55_x'])
                body['max_45_55_y'].append(features_45_55_y['max_45_55_y'])
                body['max_45_55_z'].append(features_45_55_z['max_45_55_z'])

                body['min_45_55_x'].append(features_45_55_x['min_45_55_x'])
                body['min_45_55_y'].append(features_45_55_y['min_45_55_y'])
                body['min_45_55_z'].append(features_45_55_z['min_45_55_z'])

                body['max_abs_45_55_x'].append(features_45_55_x['max_abs_45_55_x'])
                body['max_abs_45_55_y'].append(features_45_55_y['max_abs_45_55_y'])
                body['max_abs_45_55_z'].append(features_45_55_z['max_abs_45_55_z'])

                body['dist_max_min_45_55_x'].append(features_45_55_x['dis_max_min_45_55_x'])
                body['dist_max_min_45_55_y'].append(features_45_55_y['dis_max_min_45_55_y'])
                body['dist_max_min_45_55_z'].append(features_45_55_z['dis_max_min_45_55_z'])

                body['mean_45_55_x'].append(features_45_55_x['mean_45_55_x'])
                body['mean_45_55_y'].append(features_45_55_y['mean_45_55_y'])
                body['mean_45_55_z'].append(features_45_55_z['mean_45_55_z'])

                body['total_accelerated_45_55_xyz'].append(np.sqrt(
                    (features_45_55_x['mean_45_55_x'] ** 2) + (features_45_55_y['mean_45_55_y'] ** 2) + (
                            features_45_55_z['mean_45_55_z'] ** 2)))

                body['median_45_55_x'].append(features_45_55_x['median_45_55_x'])
                body['median_45_55_y'].append(features_45_55_y['median_45_55_y'])
                body['median_45_55_z'].append(features_45_55_z['median_45_55_z'])

                body['variance_45_55_x'].append(features_45_55_x['variance_45_55_x'])
                body['variance_45_55_y'].append(features_45_55_y['variance_45_55_y'])
                body['variance_45_55_z'].append(features_45_55_z['variance_45_55_z'])

                body['std_45_55_x'].append(features_45_55_x['std_45_55_x'])
                body['std_45_55_y'].append(features_45_55_y['std_45_55_y'])
                body['std_45_55_z'].append(features_45_55_z['std_45_55_z'])

                body['sum_45_55_x'].append(features_45_55_x['sum_45_55_x'])
                body['sum_45_55_y'].append(features_45_55_y['sum_45_55_y'])
                body['sum_45_55_z'].append(features_45_55_z['sum_45_55_z'])

                # general statistics for window 48 - 52

                body['max_48_52_x'].append(features_48_52_x['max_48_52_x'])
                body['max_48_52_y'].append(features_48_52_y['max_48_52_y'])
                body['max_48_52_z'].append(features_48_52_z['max_48_52_z'])

                body['min_48_52_x'].append(features_48_52_x['min_48_52_x'])
                body['min_48_52_y'].append(features_48_52_y['min_48_52_y'])
                body['min_48_52_z'].append(features_48_52_z['min_48_52_z'])

                body['max_abs_48_52_x'].append(features_48_52_x['max_abs_48_52_x'])
                body['max_abs_48_52_y'].append(features_48_52_y['max_abs_48_52_y'])
                body['max_abs_48_52_z'].append(features_48_52_z['max_abs_48_52_z'])

                body['dist_max_min_48_52_x'].append(features_48_52_x['dis_max_min_48_52_x'])
                body['dist_max_min_48_52_y'].append(features_48_52_y['dis_max_min_48_52_y'])
                body['dist_max_min_48_52_z'].append(features_48_52_z['dis_max_min_48_52_z'])

                body['mean_48_52_x'].append(features_48_52_x['mean_48_52_x'])
                body['mean_48_52_y'].append(features_48_52_y['mean_48_52_y'])
                body['mean_48_52_z'].append(features_48_52_z['mean_48_52_z'])

                # calculate the total acceleration between the 48-52 seconds
                body['total_accelerated_48_52_xyz'].append(np.sqrt(
                    (features_48_52_x['mean_48_52_x'] ** 2) + (features_48_52_y['mean_48_52_y'] ** 2) + (
                            features_48_52_z['mean_48_52_z'] ** 2)))

                body['median_48_52_x'].append(features_48_52_x['median_48_52_x'])
                body['median_48_52_y'].append(features_48_52_y['median_48_52_y'])
                body['median_48_52_z'].append(features_48_52_z['median_48_52_z'])

                body['variance_48_52_x'].append(features_48_52_x['variance_48_52_x'])
                body['variance_48_52_y'].append(features_48_52_y['variance_48_52_y'])
                body['variance_48_52_z'].append(features_48_52_z['variance_48_52_z'])

                body['std_48_52_x'].append(features_48_52_x['std_48_52_x'])
                body['std_48_52_y'].append(features_48_52_y['std_48_52_y'])
                body['std_48_52_z'].append(features_48_52_z['std_48_52_z'])

                body['sum_48_52_x'].append(features_48_52_x['sum_48_52_x'])
                body['sum_48_52_y'].append(features_48_52_y['sum_48_52_y'])
                body['sum_48_52_z'].append(features_48_52_z['sum_48_52_z'])

                # if the specific axi crossed the given threshold
                body['is_grater08_abs_x'].append(is_grater08_abs_x)
                body['is_grater08_abs_y'].append(is_grater08_abs_y)
                body['is_grater18_abs_z'].append(is_grater18_abs_z)

                # check for spikes by threshold
                body['th_05_more_then_5_spikes_55_76_x'].append(1 if th_05_spikes_55_76_x > 5 else 0)
                body['th_05_more_then_10_spikes_55_76_x'].append(1 if th_05_spikes_55_76_x > 10 else 0)
                body['th_05_more_then_5_spikes_55_76_z'].append(1 if th_25_spikes_55_76_z > 5 else 0)
                body['th_05_more_then_05_x_and_th_25_more_then_05_z_spikes_55_76'].append(
                    1 if (th_05_spikes_55_76_x > 5 and th_25_spikes_55_76_z > 5) else 0)
                body['th_05_more_then_10_x_and_th_25_more_then_05_z_spikes_55_76'].append(
                    1 if (th_05_spikes_55_76_x > 10 and th_25_spikes_55_76_z > 5) else 0)

                # number of spikes with different thresholds
                body['th_05_spikes_55_76_x'].append(th_05_spikes_55_76_x)
                body['th_10_spikes_55_76_x'].append(th_10_spikes_55_76_x)
                body['th_05_spikes_55_76_y'].append(th_05_spikes_55_76_y)
                body['th_10_spikes_55_76_y'].append(th_10_spikes_55_76_y)
                body['th_05_spikes_55_76_z'].append(th_05_spikes_55_76_z)
                body['th_10_spikes_55_76_z'].append(th_10_spikes_55_76_z)
                body['th_15_spikes_55_76_z'].append(th_15_spikes_55_76_z)
                body['th_25_spikes_55_76_z'].append(th_25_spikes_55_76_z)

                # get the number of spikes from window
                # Define the threshold for 0.5 G-force
                threshold = 0.5

                # Calculate the spikes with the 0.5 G-force threshold
                spikes_55_76_x = local_spike_counter_from_window(window_55_76_x, threshold)
                spikes_55_76_z = local_spike_counter_from_window(window_55_76_z, threshold)

                # Append the result to the body based on the spike condition
                body['spikes_55_76'].append(1 if (spikes_55_76_x > 10 and spikes_55_76_z > 10) else 0)
                # # Convert lists to numpy arrays for efficient computation
                # window_55_76_x = np.array(window_55_76_x)
                # window_55_76_y = np.array(window_55_76_y)
                # window_55_76_z = np.array(window_55_76_z)
                #
                # # Calculate the differences between consecutive values
                # diffs_x = np.abs(np.diff(window_55_76_x))
                # diffs_y = np.abs(np.diff(window_55_76_y))
                # diffs_z = np.abs(np.diff(window_55_76_z))
                #
                # # Count spikes where the difference exceeds the threshold (0.5 G-Force)
                # spikes_55_76_x = np.sum(diffs_x > 0.5)
                # spikes_55_76_y = np.sum(diffs_y > 0.5)
                # spikes_55_76_z = np.sum(diffs_z > 0.5)
                #
                # # Determine if there are more than 10 spikes in any axis
                # spikes_55_76 = 1 if (spikes_55_76_x > 10 or spikes_55_76_y > 10 or spikes_55_76_z > 10) else 0
                #
                # # Append the spikes_55_76 value to the body dictionary
                # body['spikes_55_76'].append(spikes_55_76)

        # Create Data frame - Pass in Results List & Define Column Names
        results_df = pd.DataFrame.from_dict(body)

        # Save results to CSV
        dataset_folder_path = 'data/dataset'
        dataset_folder_path = os.path.join(dataset_folder_path, out_filename)
        results_df.to_csv(f'{dataset_folder_path}.csv')

        print(body.keys())



if __name__ == '__main__':
    # json_to_csv_conv('data_by_months')
    # create_143_table_from_json('data_by_months', '143_table')
    dataset_creator('data_by_months', 'features_31-08-24_test', '143_table')
