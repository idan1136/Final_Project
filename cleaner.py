import math
import argparse
import os
import json
from utiles import feature_extractor, json_serializer
from tqdm import tqdm


def all_cases_to_txt_file(path, output_filename):
    """ Function that get a path to a folder contains JSON files and delete unnecessary files

                :param path: valid path to folder with JSON accidents files
                :type path: String

                """
    # counters that will tell us how much cases we deleted
    acc_del, not_acc_del = 0, 0

    # go over all the folders in directory (accident and not accident)
    for folder_name in tqdm(os.listdir(path), desc="Processing folders", unit="folder"):

        # "0" if it's the not_accidents folders and "1" if it's the real accidents folder
        is_pos = True if folder_name.startswith('1') else False

        # directory to specific folder
        directory = f'{path}/{folder_name}'

        # Iterate over all files inside the given folder
        for filename in tqdm(os.listdir(directory), desc="Processing files", unit="file"):
            # Get json file path
            json_file_path = os.path.join(directory, filename)

            # Opening JSON file
            with open(json_file_path, encoding="utf-8") as f:
                json_data = json.load(f)


            # translate prediction
            label_class = 'Accident' if is_pos else 'Not Accident'

            # Extract  date, ip, ari and description
            payload = json_data['payloads'][0].split(',')
            date = payload[6]
            ip = json_data["IP"]

            # check if there is ARI and Description
            description = json_data['description'] if 'description' in json_data else 'No Description'
            ari = json_data['ari_link'] if 'ari_link' in json_data else 'No ARI Link'


            # Write it to a text file
            with open(f'./results/{output_filename}.txt', 'a+', encoding='utf-8') as f:
                f.write(
                    'Class : ' + str(label_class) +
                    '\nFile Name : ' + filename +
                    '\nIP : ' + ip +
                    '\nDate : ' + date +
                    '\n' + description +
                    '\nARI : ' + ari +
                    '\n\n\n')


def filter_labeled_data_by_quality_check(path):
    """ Function that get a path to a folder contains JSON files and delete unnecessary files

            :param path: valid path to folder with JSON accidents files
            :type path: String

            """
    # counters that will tell us how much cases we deleted
    acc_del, not_acc_del = 0, 0

    # go over all the folders in directory (accident and not accident)
    for folder_name in tqdm(os.listdir(path), desc="Checking folders", unit="folder"):

        # "0" if it's the not_accidents folders and "1" if it's the real accidents folder
        is_pos = True if folder_name.startswith('1') else False

        # directory to specific folder
        directory = f'{path}/{folder_name}'

        # Iterate over all files inside the given folder
        for filename in tqdm(os.listdir(directory), desc="Checking files", unit="file"):

            # Get json file path
            json_file_path = os.path.join(directory, filename)

            # Opening JSON file
            with open(json_file_path, encoding="utf-8") as f:
                json_data = json.load(f)

            # Preprocessing json string
            json_str = json_serializer(json_data)
            try:
                # extract feature from json
                features = feature_extractor(json_str)

            except Exception as e:
                print(f'Error for: {filename}\Exception: {e}')
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)
                if is_pos:
                    acc_del += 1
                else:
                    not_acc_del += 1
                continue

    print(f'Accidents deleted: {acc_del}\nNot Accidents deleted: {not_acc_del}')


def filter_labeled_data_by_desc(path):
    """ Function that get a path to a folder contains JSON files and delete unnecessary files with no description

        :param path: valid path to folder with JSON accidents files
        :type path: String

        """
    # counters that will tell us how much cases we deleted
    acc_del, not_acc_del = 0, 0

    # go over all the folders in directory (accident and not accident)
    for folder_name in tqdm(os.listdir(path), desc="Filtering by description", unit="folder"):

        # "0" if it's the not_accidents folders and "1" if it's the real accidents folder
        is_pos = True if folder_name.startswith('1') else False

        # directory to specific folder
        directory = f'{path}/{folder_name}'

        # Iterate over all files inside the given folder
        for filename in tqdm(os.listdir(directory), desc="Filtering files", unit="file"):

            # Get json file path
            json_file_path = os.path.join(directory, filename)

            # Opening JSON file
            with open(json_file_path, encoding="utf-8") as f:
                json_data = json.load(f)

            try:
                if not json_data['description']:
                    if os.path.exists(json_file_path):
                        os.remove(json_file_path)
                        if is_pos:
                            acc_del += 1
                        else:
                            not_acc_del += 1

                        continue
            except Exception as e:
                print("description", e, '\n', json_file_path)
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)
                if is_pos:
                    acc_del += 1
                else:
                    not_acc_del += 1

                continue

    print(f'Accidents deleted: {acc_del}\nNot Accidents deleted: {not_acc_del}')


def parser_args():
    parser = argparse.ArgumentParser(description='Calculate volume of cv')
    parser.add_argument('--r', type=int, help='Radius of Cylinder')
    parser.add_argument('--h', type=int, help='height of Cylinder')
    return parser.parse_args()


def cv(r, h):
    v = (math.pi) * ( r ** 2) * h
    return v

def show_erez_os():
    folder_path = 'data\dataset'
    folder_names = os.listdir(folder_path)
    print(folder_names)

    for filename in folder_names:
        file_path = os.path.join(folder_path, filename)
        print(file_path)


def clean_all(path):
    filter_labeled_data_by_desc(path)
    filter_labeled_data_by_quality_check(path)


def delete_all_from_txt(txt_file_path, delete_file_path):
    """

    :param txt_file_path: The path for the text file that contains the files names for delete
    :type txt_file_path: string
    :param delete_file_path: The path that contains the folders of data we want to delete from
    :type delete_file_path: string
    """

    # read filenames from text file
    with open(txt_file_path) as f:
        file_names_for_delete = [line.strip() for line in f.readlines()]

    # counters that will tell us how much cases we deleted
    acc_del, not_acc_del = 0, 0

    # go over all the folders in directory (accident and not accident)
    for folder_name in tqdm(os.listdir(delete_file_path), desc="Deleting from folders", unit="folder"):

        # "0" if it's the not_accidents folders and "1" if it's the real accidents folder
        is_pos = True if folder_name.startswith('1') else False

        # directory to specific folder
        directory = f'{delete_file_path}/{folder_name}'

        # Iterate over all files inside the given folder
        for filename in tqdm(os.listdir(directory), desc="Deleting files", unit="file"):

            # Get json file path
            json_file_path = os.path.join(directory, filename)

            # run over all the file name and delete the names we find in the list
            if filename in file_names_for_delete:
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)
                    if is_pos:
                        acc_del += 1
                    else:
                        not_acc_del += 1

    print(f'Accidents deleted: {acc_del}\nNot Accidents deleted: {not_acc_del}')




if __name__ == '__main__':
    clean_all('data_by_months')
    args = parser_args()
    print(cv(args.r, args.h))
    all_cases_to_txt_file("filterd_labeled_data", "train")
    delete_all_from_txt(txt_file_path ="errors/ .txt", delete_file_path="filterd_labeled_data")
