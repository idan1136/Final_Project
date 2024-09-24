import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime as dt



def json_to_dictionary (path_to_json : str)->dict:

    """
    this function will ask from the user path to the Json file and will return it as a dict.
    :raise value error: if path not exists
    :param path_to_json:  path to the json file.

    :return: dict of the python file
    """
    if not os.path.exists(path_to_json):
        raise ValueError(f'{path_to_json} -path not exists')
    # Open the JSON file with the correct encoding
    with open( path_to_json, encoding='utf-8') as f:
        # Load the JSON data from the file
        json_data = json.load(f)

    #arange the dict to lines
    # json_data = json.dumps(json_data, indent=4)
    return json_data


def dict_to_axis_dict(path_to_json: str)->dict:
    """
    The function divides the file into 3 axes (X, Y, Z)

    :param path_to_json:Receives from the main function the file as dictionary

    :return: the graph of the axes
    """
    json_dict = json_to_dictionary(path_to_json)

    # create lists for axis
    x_axi, y_axi, z_axi = [], [], []

    #this loop create a long list with ',' instend of ';'
    for second_in_payload in json_dict['payloads']:
        split_list = second_in_payload.replace(';', ',')
        split_list = split_list.split(',')[11:]

    #this loop save the values in axis
        for index in range(0, len(split_list), 3):
                x_axi.append(float(split_list[index]))
                y_axi.append(float(split_list[index + 1]))
                z_axi.append(float(split_list[index + 2]))


    return {
        'x_axi':x_axi,
        'y_axi':y_axi,
        'z_axi':z_axi
    }

def show_graph(path_to_json:str, plot_path:str = None, is_save:bool = True )->None:
    """


    :param axis_dict: axis dict
    :param plot_name the name of the plot
    :param is_save variable that determine if the plot will be save or only show

    :return: void
    """

#we change the end of the file to png if the plot name is none than its will contain the same name as the json file

    if plot_path is None:
        plot_path = path_to_json
        plot_path.replace('json', 'png')
    else:
        plot_path = plot_path+'.png'


    axis_dict = dict_to_axis_dict(path_to_json)

    #creates the "base" of the graph
    fig, ax = plt.subplots(3, 1, figsize=(15, 6))

    #creates a list to the axis and to the colors
    data = [axis_dict['x_axi'], axis_dict['y_axi'],axis_dict[ 'z_axi']]
    colors = ['green', 'red', 'blue']

    #show the graph for the axis in for loop
    for i in range(3):
        ax[i].plot(data[i], color=colors[i])
        ax[i].set_xlabel('Index')
        ax[i].set_ylabel(['x-axis values', 'y-axis values', 'z-axis values'][i])

    if is_save is True:
        plt.savefig(plot_path)
    else:
        plt.show()

    plt.close()
def variable_checker(curr_var):
    if curr_var == None:
        return 'None'
    else:
        return curr_var
def jsons_to_csv(path_to_folder:str)-> None:
    """
    this function will ask path to the folder that contain the accident or not accident folder


    :param path_to_folder: path to the folder that contain the accident or not accident folder
    :return:None

    """

    col_file_name = []
    col_primary_label = []
    col_description = []
    col_accident_direction = []
    col_ari_link = []
    col_lim_143 = []
    col_date = []
    col_ip = []
    col_pocsag = []


    #axtrcat all the folders ( accident or not accident )
    folder_name_list = os.listdir(path_to_folder)
    for folder_name in folder_name_list:
        #determine the primary libary of the name in the folder
        if folder_name  == 'accidents':
            primary_label = 1
        else:
            primary_label = 0

        #concatinate the path to the folder
        folder_path = os.path.join(path_to_folder, folder_name)

        # extract all the json files names from the folder path
        json_name_list  = os.listdir(folder_path)

        #iterate over all the json file from the list

        for json_name in json_name_list:
            json_path = os.path.join(folder_path,json_name)
            json_data = json_to_dictionary(json_path)
            col_file_name.append(json_name)
            col_primary_label.append(primary_label)

            # extract json description
            col_description.append(variable_checker(json_data['description']))

            # extract json accident direction
            try:
                col_accident_direction.append(variable_checker(json_data['accident_direction']))
            except:
                col_accident_direction.append('None')

            try:
                col_ari_link.append(variable_checker(json_data['ari_link']))
            except:
                col_ari_link.append('None')

            #take the last value of lim_143
            curr_lim_143 = json_data['lim_143']
            curr_lim_143 = curr_lim_143.split(',')[-1]
            col_lim_143.append(curr_lim_143)

            col_date.append(json_data['date'])
            col_ip.append(json_data['IP'])
            col_pocsag.append(json_data['pocsag'])

            plot_name = json_name
            plot_name = plot_name.replace('.json','')

            plot_path =f'./plots/{folder_name}/{plot_name}'

            show_graph(path_to_json=json_path, plot_path=plot_path)

    dict_data= {
        'File Name' : col_file_name,
        'Primary label': col_primary_label,
        'Description' : col_description,
        'Accident direction' : col_accident_direction,
        'Ari_link' : col_ari_link,
        'Lim_143' : col_lim_143,
        'Date' : col_date,
        'Ip' : col_ip,
        'Pocsag' : col_pocsag

    }
    df = pd.DataFrame(dict_data)
    today_str = dt.now().strftime("%d-%m-%Y")
    df.to_csv(f'./labeling_file/{today_str}.csv',encoding='utf-8-sig')



def find_stat_axis_to_labelig_file(axis_list):
    """ function that return the x and the y values where x had the maximum value """

    # create 3 list for the axis and a time list for the time line of their incomes
    x_list = []
    y_list = []
    z_list = []

    # three more lists for the correlations
    cx_list = []
    cy_list = []
    cz_list = []

    for xyz in axis_list:
        # split the the parameters and then casting them to float
        xyz_split = xyz.split(',')

        cx_list.append(float(xyz_split[0]))
        if float(xyz_split[0]) != 0.0:
            x_list.append(float(xyz_split[0]))

        cy_list.append(float(xyz_split[1]))
        if float(xyz_split[1]) != 0.0:
            y_list.append(float(xyz_split[1]))

        cz_list.append(float(xyz_split[2]))
        if float(xyz_split[2]) != 1.0:
            z_list.append(float(xyz_split[2]))

        # return statistics

    # find correlation between the axis
    df = pd.DataFrame()
    df['x'] = cx_list
    df['y'] = cy_list
    df['z'] = cz_list

    xyz_per_sec = len(cx_list)/76

    # calculate how much samples every milisec
    sample_every_milisec = 1000 / xyz_per_sec

    x_above_th_in_milisec = 0
    y_above_th_in_milisec = 0
    z_above_th_in_milisec = 0

    for i in range(len(cx_list)):
        if abs(cx_list[i]) > 1.2:
            x_above_th_in_milisec += 1
        if abs(cy_list[i]) > 0.9:
            y_above_th_in_milisec += 1
        if abs(cz_list[i]) > 2.0:
            z_above_th_in_milisec += 1

    df['x_above_th_in_milisec'] = x_above_th_in_milisec * sample_every_milisec
    df['y_above_th_in_milisec'] = y_above_th_in_milisec * sample_every_milisec
    df['z_above_th_in_milisec'] = z_above_th_in_milisec * sample_every_milisec

    try:
        my_stat = {
            'x_above_th_in_milisec': (x_above_th_in_milisec * sample_every_milisec),
            'y_above_th_in_milisec': (y_above_th_in_milisec * sample_every_milisec),
            'z_above_th_in_milisec': (z_above_th_in_milisec * sample_every_milisec),
            'x_max': max(x_list),
            'y_max': max(y_list),
            'z_max': max(z_list),
            'x_min': min(x_list),
            'y_min': min(y_list),
            'z_min': min(z_list)
        }
        return my_stat

    except:

        return {
            'x_above_th_in_milisec': '##',
            'y_above_th_in_milisec': '##',
            'z_above_th_in_milisec': '##',
            'x_max': '##',
            'y_max': '##',
            'z_max': '##',
            'x_min': '##',
            'y_min': '##',
            'z_min': '##'
        }

def add_values_to_labeling_file(labeling_file_name):
    """ for each trip and unify its dismantled maneuvers """

    count = 0

    df = pd.read_csv(f'./labeling_file/{labeling_file_name}.csv')

    pocsags = [float(pl) for pl in df['Pocsag']]
    x_max_lst, y_max_lst, z_max_lst = ['$$' for _ in range(len(pocsags))], ['$$' for _ in range(len(pocsags))], ['$$' for _ in range(len(pocsags))]
    x_min_lst, y_min_lst, z_min_lst = ['$$' for _ in range(len(pocsags))], ['$$' for _ in range(len(pocsags))], ['$$' for _ in range(len(pocsags))]
    x_above_lst, y_above_lst, z_above_lst = ['$$' for _ in range(len(pocsags))], ['$$' for _ in range(len(pocsags))], ['$$' for _ in range(len(pocsags))]

    for index_in_df, row in df.iterrows():
        if row['Primary label'] == 1:
            folder_name = 'accidents'
        else:
            folder_name = 'not_accidents'

        file_name = row['File Name']




        # reading json
        file_path = f'./data_by_months/{folder_name}/{file_name}'
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        #plate of the json file
        file_pocsag = float(data['pocsag'])
        try:
            file_ari = str(data['ari_link'])
        except:
            count +=1
            file_ari = row['Ari_link']

        # get index of json file's plate in table
        #index_in_df = df[(df['Pocsag'] == file_pocsag) & (df['Ari_link'] == file_ari)].index.values[0]

        # list of payloads of the whole trip
        payloads = data['payloads']


        axis_list = []

        for idx, p in enumerate(payloads):

            # get the axis
            axis = p.split(";")
            tmp = axis[0].split(',')
            if idx == 51:
                accident_time = tmp[6]
            axis[0] = ','.join(tmp[-3:])
            axis_list.extend(axis)

        clean_accident = axis_list

        #pocsag not in labeling csv file
        if file_pocsag not in pocsags:
            print(f'{file_pocsag} not in the table')
            continue
        acc_stat = find_stat_axis_to_labelig_file(clean_accident)
        x_max, y_max, z_max = acc_stat['x_max'], acc_stat['y_max'], acc_stat['z_max']
        x_min, y_min, z_min = acc_stat['x_min'], acc_stat['y_min'], acc_stat['z_min']
        x_above, y_above, z_above = acc_stat['x_above_th_in_milisec'], acc_stat['y_above_th_in_milisec'], acc_stat['z_above_th_in_milisec']

        x_max_lst[index_in_df] = x_max
        y_max_lst[index_in_df] = y_max
        z_max_lst[index_in_df] = z_max
        x_min_lst[index_in_df] = x_min
        y_min_lst[index_in_df] = y_min
        z_min_lst[index_in_df] = z_min
        x_above_lst[index_in_df] = x_above
        y_above_lst[index_in_df] = y_above
        z_above_lst[index_in_df] = z_above

    df['X Max'] = x_max_lst
    df['X Min'] = x_min_lst
    df['Y Max'] = y_max_lst
    df['Y Min'] = y_min_lst
    df['Z Max'] = z_max_lst
    df['Z Min'] = z_min_lst

    df['X Above 1.2 (Milisec)'] = x_above_lst
    df['Y Above 0.9 (Milisec)'] = y_above_lst
    df['Z Above 2.0 (Milisec)'] = z_above_lst

    df.to_csv(f'./labeling_file_with_values/{labeling_file_name}.csv',encoding='utf-8-sig')
    print(f'Errors: {count}')

if __name__ == '__main__' :
    #show_graph('case1.json')
    #jsons_to_csv('data_by_months')
    add_values_to_labeling_file('08-05-2023')



