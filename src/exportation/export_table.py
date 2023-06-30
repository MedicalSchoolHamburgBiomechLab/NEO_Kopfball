import os.path
from src.constants import *
from src.constants.calculated_constants import *
import utils as utils
import openpyxl
import datetime
from concurrent.futures import ProcessPoolExecutor


def row_task(row, task):
    value_subject = [row['ID']]

    try:
        df_task = pd.read_csv(os.path.join(
            os.path.dirname(utils.get_files(PATH_PROJECT, extension=row['session_1'] + ".eyetrax")[0]), task,
            task + "_outcomes.csv"))

        for outcomes in CONST_PARAMETERS[task][2]:
            # for block with one value save as str
            try:
                value_subject.append(float(df_task.PARAMETERS_VALUES[df_task.SACCADE_PARAMETERS == outcomes].values))
            except:
                #for block with array of values save as str
                try:
                    tab_values = np.array([float(x) for x in df_task.PARAMETERS_VALUES[df_task.SACCADE_PARAMETERS == outcomes].values[0][1:-1].split(',')])
                    tab_values_filtered = tab_values[~np.isnan(tab_values)]
                    value_subject.append(np.median(tab_values_filtered))
                except:
                    value_subject.append(np.nan)

    except Exception as exception:
        for nan_value in range(len(CONST_PARAMETERS[task][2])):
            value_subject.append(np.nan)

    try:
        df_task = pd.read_csv(os.path.join(
            os.path.dirname(utils.get_files(PATH_PROJECT, extension=row['session_2'] + ".eyetrax")[0]), task,
            task + "_outcomes.csv"))
        for outcomes in CONST_PARAMETERS[task][2]:
            #for block with one value save as str
            try:
                value_subject.append(float(df_task.PARAMETERS_VALUES[df_task.SACCADE_PARAMETERS == outcomes].values))
            except:
                #for block with array of values save as str
                try:
                    tab_values = np.array([float(x) for x in df_task.PARAMETERS_VALUES[df_task.SACCADE_PARAMETERS == outcomes].values[0][1:-1].split(',')])
                    tab_values_filtered = tab_values[~np.isnan(tab_values)]
                    value_subject.append(np.median(tab_values_filtered))
                except:
                    value_subject.append(np.nan)

    except Exception as exception:
        for nan_value in range(len(CONST_PARAMETERS[task][2])):
            value_subject.append(np.nan)

    try:
        df_task = pd.read_csv(os.path.join(
            os.path.dirname(utils.get_files(PATH_PROJECT, extension=row['session_3'] + ".eyetrax")[0]), task,
            task + "_outcomes.csv"))
        for outcomes in CONST_PARAMETERS[task][2]:
            #for block with one value save as str
            try:
                value_subject.append(float(df_task.PARAMETERS_VALUES[df_task.SACCADE_PARAMETERS == outcomes].values))
            except:
                #for block with array of values save as str
                try:
                    tab_values = np.array([float(x) for x in df_task.PARAMETERS_VALUES[df_task.SACCADE_PARAMETERS == outcomes].values[0][1:-1].split(',')])
                    tab_values_filtered = tab_values[~np.isnan(tab_values)]
                    value_subject.append(np.median(tab_values_filtered))
                except:
                    value_subject.append(np.nan)

    except Exception as exception:
        for nan_value in range(len(CONST_PARAMETERS[task][2])):
            value_subject.append(np.nan)
    return task, value_subject


def main():
    df = pd.read_excel(os.path.join(PATH_PROJECT, 'resources', CONST_STRAIGHT))
    df = df.sort_values('ID')
    list_sheet_names = list(CONST_PARAMETERS.keys())
    table_session = ['session1', 'session2', 'session3']
    # Create a workbook and add a worksheet.
    wb = openpyxl.Workbook()
    for task in list_sheet_names:
        wb.create_sheet(task)
        tab_header = ['ID']
        for j in range(len(table_session)):
            for i in range(len(CONST_PARAMETERS[task][2])):
                tab_header.append('_'.join([table_session[j],CONST_PARAMETERS[task][2][i]]))
        wb[task].append(tab_header)

    # TODO:
    # # OPTIMIZE PROCESS
    # with ProcessPoolExecutor(utils.get_cpu_core_count()) as executor:
    #     for index, row in tqdm(df.iterrows()):
    #         for task in list_sheet_names:
    #             executor_process = executor.submit(main, row=row, task=task)
    #             result_executor = executor_process.result()
    #             wb[result_executor[0]].append(result_executor[1])

    for index, row in df.iterrows():
        print(row)
        for task in list_sheet_names:
            result_executor = row_task(row, task)
            tab = [x if not pd.isna(x) else 'nan' for x in result_executor[1]]
            wb[result_executor[0]].append(tab)



    wb.remove(wb['Sheet'])
    wb.save(os.path.join(PATH_PROJECT, 'result_analysis',
                         f'Table_result_{CONST_DATA_TYPE}_{datetime.datetime.now().strftime("%b_%d_%Y_(%Hh_%Mm)")}.xlsx'))
    wb.close()