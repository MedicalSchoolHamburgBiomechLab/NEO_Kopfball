from src.constants import *
from src.constants import constants_processing
from src.constants import constants_check

import os
from concurrent.futures import ProcessPoolExecutor
from queue import Queue
from tqdm import tqdm
import utils

# Init queue error
queue = Queue()


def add_error(result):
    try:
        queue.put(result[0])
        queue.put(result[1])
        queue.put("_____________________________")
    except:
        pass


def save_error(queue_values):
    with open(os.path.join(PATH_PROJECT, 'result_analysis', f'error_{CONST_DATA_TYPE}.txt'), 'w') as f:
        for item in queue_values.queue:
            if len(item) > 0:
                f.write(item)
                f.write("\n")


def constants_part():
    constants_processing.main()
    # constants_check.main()


def processing_part():
    # ___________________________Settings function___________________________

    # [name_task]: <str> name of a specific task (==> [biflicker, vertical biflicker, sine, circle, velo horizontal,
    # velo vertical, antibiflicker]),  default=every task

    # outcomes_calculation: <bool> boolean that calculated the outcomes,    default=True

    # [path_subject]: <str> eyetrax file of a specific subject,   default=every file ".eyetrax"

    # save_plot: <bool> boolean to save or not the plot of the task,    default=True

    # global_plot: <bool> boolean to save or not the global plot of the task of every subject,    default=False

    # saccade_detection: <bool> boolean to run the saccade/fixation algorithm,   default=True

    # save_data_processed: <bool> boolean that save the final dataFrame of the task in pickle,  default=True

    # ______________________________________________________________________

    list_path = utils.get_files(os.path.join(PATH_PROJECT, 'data', CONST_DATA), extension=".eyetrax")
    # list_path = [r"C:\Users\william.bouchut\Documents\Work\Neo_Kopfball\data\new\S3\MSH_Hauptstudie_S3\MSH_Kopfball_S3\2022-10-21\9ca5e47e-e608-44e7-9acb-cb2c3ad58d70\9ca5e47e-e608-44e7-9acb-cb2c3ad58d70.eyetrax"]

    # OPTIMIZE PROCESS
    with ProcessPoolExecutor(utils.get_cpu_core_count()) as executor:
        for path in tqdm(list_path):
            future = executor.submit(data_processing.main, path_subject=[path], outcomes_calculation=True,
                                     saccade_detection=False, save_plot=False, save_data_processed=False,
                                     global_plot=False)
            future.add_done_callback(lambda x: add_error(x.result()))

    save_error(queue)


def exportation_part():
    export_table.main()


if __name__ == '__main__':

    print('-----CONSTANTS PART-----')
    # constants_part()

    from src.processing import data_processing

    print('-----DATA PROCESSING PART-----')
    processing_part()

    from src.exportation import export_table

    print('-----EXPORTATION PART-----')
    exportation_part()

    print('-----PROCESS FINISHED-----')
