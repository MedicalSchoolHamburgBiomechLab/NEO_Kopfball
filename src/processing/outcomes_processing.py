from src.constants import *
from src.constants.calculated_constants import *

import pandas as pd
import utils as utils
import numpy as np
import csv
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def check_shape_consistency(df_saccade, name_task):
    # tab_speed = []
    # # TODO: add rate in the derivation calculation
    # tab_speed = np.diff(df_saccade[CONST_PARAMETERS[name_task][0][0]])
    # sign_check = tab_speed > 0
    # if not np.allclose(sign_check, sign_check[0]):
    #
    #     # Check that the variation of the eye is coherent by checking that the number of extrem variation inside the
    #     # graph is not above 3
    #     if sum(tab_speed > (0.7 * np.max(np.abs(tab_speed)))) > 3:
    #         return False
    # else:
    #     return True
    return True


def check_anticipated_saccade(df, task_timestamp):
    df['Anticipated_Saccade'] = np.nan

    for i in range(len(task_timestamp) - 1):

        df_saccade = df[
            df.timestamp.between(task_timestamp[i] - CONST_SACCADE['window_first_saccade'],
                                 task_timestamp[i + 1]) & (df.Valid_Saccade == True)]
        if not df_saccade.empty:
            df_first_saccade = df_saccade[df_saccade.NSLR_Segment == df_saccade.NSLR_Segment.iloc[0]]
            if df_first_saccade.timestamp.iloc[0] < task_timestamp[i]:
                df['Anticipated_Saccade'].loc[df_first_saccade.index] = True

    return df


def check_saccade_consistency(df, df_ball_position, path_saccade_analysis, name_task):
    list_saccade_index = df.NSLR_Class[df.NSLR_Class == 'Saccade'].index.values
    group_saccade = np.split(list_saccade_index, np.where(np.diff(list_saccade_index) != 1)[0] + 1)
    # group_saccade_filtered = [list(group_saccade[0])]
    group_saccade_filtered = [group_saccade[0]]
    df['Valid_Saccade'] = np.nan
    Uncorrect_Saccade = 0

    # check that saccade is bounded by two fixation
    for i in range(0, len(group_saccade) - 1):
        if not 'Fixation' in df.NSLR_Class.loc[group_saccade[i][-1]:group_saccade[i + 1][0]].values:
            group_saccade_filtered[-1] = np.concatenate((group_saccade_filtered[-1], group_saccade[i + 1]))
        else:
            group_saccade_filtered.append(group_saccade[i + 1])

    for saccade in group_saccade_filtered:
        df.NSLR_Segment.loc[saccade] = df.NSLR_Segment.loc[saccade[0]]
        df_saccade = df.loc[saccade[0]:saccade[-1]]

        # Check the duration of the saccade
        if CONST_SACCADE['max_timestamp'] <= (saccade[-1] - saccade[0]) >= CONST_SACCADE['min_timestamp']:
            if check_shape_consistency(df_saccade=df_saccade, name_task=name_task):
                df.Valid_Saccade.loc[saccade] = True
            else:
                df.Valid_Saccade.loc[saccade] = False
        else:
            df.Valid_Saccade.loc[saccade] = False
            Uncorrect_Saccade += 1

    saccade_analysis = {
        'Number Saccade': len(group_saccade),
        'Valid Saccade': len(group_saccade_filtered) - Uncorrect_Saccade,
    }

    # SAVE RESULT IN CSV FILE
    with open(os.path.join(os.path.dirname(path_saccade_analysis), name_task.replace(" ", ""),
                           name_task + "_saccade_analysis.csv"),
              'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['SACCADE_PARAMETERS', 'PARAMETERS_VALUES'])
        writer.writerow(['Task_Name', name_task])
        for key, value in saccade_analysis.items():
            writer.writerow([key, value])

    # df['Valid_Saccade'] = True1

    return df


def get_df_saccade_valide(df, df_ball_position, task_timestamp, name_task):
    tab_df = []
    for i in range(1, len(task_timestamp) - 1):
        # Check if the task_timestamp is a stimulus or the centering of the ball
        value_test = (df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])][
                          df_ball_position.timestamp.between(task_timestamp[i], task_timestamp[i + 1])] == 0).values

        if np.count_nonzero(value_test == True) < np.count_nonzero(value_test == False):
            df_saccade = df[
                df.timestamp.between(task_timestamp[i] - CONST_SACCADE['window_first_saccade'],
                                     task_timestamp[i + 1]) & (df.Valid_Saccade == True) & (
                        df.Valid_direction == True)]
            if not df_saccade.empty:
                df_first_saccade = df_saccade[df_saccade.NSLR_Segment == df_saccade.NSLR_Segment.iloc[0]]

                tab_df.append(df_first_saccade)
    return tab_df


def get_saccade(df):
    tab_df = []

    list_saccade_index = df.NSLR_Class[df.NSLR_Class == 'Saccade'].index.values
    group_saccade = np.split(list_saccade_index, np.where(np.diff(list_saccade_index) != 1)[0] + 1)
    for group in group_saccade:
        if len(group) > 1:
            tab_df.append(df.loc[group])

    return tab_df


def directional_error(df, task_timestamp, df_ball_position, name_task):
    correct_direction = 0
    df['Valid_direction'] = False

    # Prosaccade
    if 'antibiflicker' in name_task:
        for i in range(1, len(task_timestamp) - 1):

            # Check if the task_timestamp is a stimulus or the centering of the ball
            value_test = (df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])][
                              df_ball_position.timestamp.between(task_timestamp[i], task_timestamp[i + 1])] == 0).values

            if np.count_nonzero(value_test == True) < np.count_nonzero(value_test == False):

                try:
                    df_saccade = df[
                        df.timestamp.between(task_timestamp[i] - CONST_SACCADE['window_first_saccade'],
                                             task_timestamp[i + 1]) & (df.Valid_Saccade == True) & (
                                df.Anticipated_Saccade != True)]
                    if not df_saccade.empty:
                        # Get the saccade direction value in the correct axis
                        direction_saccade = df_saccade[CONST_PARAMETERS[name_task][0][0]].values

                        # take the ball position angles with 0.05 before the task_timestamp in result of the between function: lead to no variation due to the use of time value
                        direction_ball = df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])][
                            df_ball_position.timestamp.between(task_timestamp[i] - 0.05,
                                                               (task_timestamp[i] + task_timestamp[i + 1]) / 2)].values

                        # Get the variation sign of the ball
                        if not np.all(np.diff(direction_ball) == 0):
                            if np.all(np.diff(direction_ball) >= 0):
                                variation_sign = 1
                            elif np.all(np.diff(direction_ball) <= 0):
                                variation_sign = -1
                            else:
                                raise TypeError("Variation of the ball not defined")

                        # Negativ sign indicating that the eye moved at the opposite direction
                        if np.diff(direction_saccade)[0] * variation_sign < 0:
                            df['Valid_direction'].loc[df_saccade.index] = True
                            correct_direction += 1

                except Exception as exception:
                    pass
                    # print(f"Directionnal error: {exception}")

    # Antisaccade
    else:
        for i in range(1, len(task_timestamp) - 1):

            # Check if the task_timestamp is a stimulus or the centering of the ball
            value_test = (df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])][
                              df_ball_position.timestamp.between(task_timestamp[i], task_timestamp[i + 1])] == 0).values

            if np.count_nonzero(value_test == True) < np.count_nonzero(value_test == False):

                try:
                    df_saccade = df[
                        df.timestamp.between(task_timestamp[i] - CONST_SACCADE['window_first_saccade'],
                                             task_timestamp[i + 1]) & (df.Valid_Saccade == True) & (
                                    df.Anticipated_Saccade != True)]

                    # Get the saccade direction value in the correct axis
                    direction_saccade = df_saccade[CONST_PARAMETERS[name_task][0][0]].values

                    # take the ball position angles with 0.05 before the task_timestamp in result of the between function: lead to no variation due to the use of time value
                    direction_ball = df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])][
                        df_ball_position.timestamp.between(task_timestamp[i] - 0.05,
                                                           (task_timestamp[i] + task_timestamp[i + 1]) / 2)].values

                    # Get the variation sign of the ball
                    if not np.all(np.diff(direction_ball) == 0):
                        if np.all(np.diff(direction_ball) >= 0):
                            variation_sign = 1
                        elif np.all(np.diff(direction_ball) <= 0):
                            variation_sign = -1
                        else:
                            raise TypeError("Variation of the ball not defined")

                    # TODO: use statistical test such as t-test

                    # Positiv sign indicating that the eye moved in the same direction
                    if np.diff(direction_saccade)[0] * variation_sign > 0:
                        df['Valid_direction'].loc[df_saccade.index] = True
                        correct_direction += 1

                except Exception as exception:
                    pass
                    # print(f"Directionnal error: {exception}")

    directional_errror = 1 - correct_direction / (len(task_timestamp) - 2)

    return directional_errror, df


def latency(df, task_timestamp, df_ball_position, name_task):
    tab_latency = []

    # Check if Prosaccade or Antisaccade
    if 'antibiflicker' in name_task:
        for i in range(1, len(task_timestamp) - 1):
            latency = np.nan

            # Check if the task_timestamp is a stimulus or the centering of the ball
            value_test = (df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])][
                              df_ball_position.timestamp.between(task_timestamp[i], task_timestamp[i + 1])] == 0).values

            if np.count_nonzero(value_test == True) < np.count_nonzero(value_test == False):

                try:
                    df_first_saccade = df[
                        df.timestamp.between(task_timestamp[i] - CONST_SACCADE['window_first_saccade'],
                                             task_timestamp[i + 1]) & (df.Valid_Saccade == True) & (
                                df.Valid_direction == True) & (df.Anticipated_Saccade != True)]

                    segment_first_saccade = df_first_saccade.NSLR_Segment.iloc[0]
                    timestamp_first_saccade = \
                        df_first_saccade.timestamp[df_first_saccade.NSLR_Segment == segment_first_saccade].iloc[0]

                    latency_stimulus = timestamp_first_saccade - task_timestamp[i]

                    if latency_stimulus > 0:
                        latency = latency_stimulus * 1000
                        tab_latency.append(latency)
                    else:
                        tab_latency.append(latency)


                except Exception as exception:
                    tab_latency.append(latency)

    else:

        for i in range(1, len(task_timestamp) - 1):

            try:
                df_first_saccade = df[
                    df.timestamp.between(task_timestamp[i] - CONST_SACCADE['window_first_saccade'],
                                         task_timestamp[i + 1]) & (df.Valid_Saccade == True) & (
                            df.Valid_direction == True) & (df.Anticipated_Saccade != True)]

                segment_first_saccade = df_first_saccade.NSLR_Segment.iloc[0]
                timestamp_first_saccade = \
                    df_first_saccade.timestamp[df_first_saccade.NSLR_Segment == segment_first_saccade].iloc[0]

                latency_stimulus = timestamp_first_saccade - task_timestamp[i]

                if latency_stimulus > 0:
                    tab_latency.append(latency_stimulus * 1000)
                else:
                    tab_latency.append(np.nan)


            except Exception as exception:
                tab_latency.append(np.nan)

    return tab_latency


def peak_velocity(df, task_timestamp, df_ball_position, name_task):
    tab_df = get_df_saccade_valide(df=df, df_ball_position=df_ball_position, task_timestamp=task_timestamp,
                                   name_task=name_task)
    tab_peak_velocity = []
    for df_saccade in tab_df:
        try:
            if df_saccade.Anticipated_Saccade.iloc[0] == True:
                tab_peak_velocity.append(np.nan)
            else:
                eye_velocity = np.diff(df_saccade[CONST_PARAMETERS[name_task][0][0]]) / np.diff(df_saccade.timestamp)
                tab_peak_velocity.append(np.max(np.abs(eye_velocity)))
        except:
            tab_peak_velocity.append(np.nan)
    return tab_peak_velocity


def gain(df, task_timestamp, df_ball_position, name_task):
    tab_df = get_df_saccade_valide(df=df, df_ball_position=df_ball_position, task_timestamp=task_timestamp,
                                   name_task=name_task)
    tab_gain = []

    gain_ball = np.abs(np.max(df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])]))

    if 'antibiflicker' not in name_task:
        gain_ball = gain_ball * 2

    for df_saccade in tab_df:

        try:
            if df_saccade.Anticipated_Saccade.iloc[0] == True:
                tab_gain.append(np.nan)
            else:
                gain = np.abs((df_saccade[CONST_PARAMETERS[name_task][0][0]].iloc[-1] -
                               df_saccade[CONST_PARAMETERS[name_task][0][0]].iloc[0])) / (gain_ball)
                tab_gain.append(gain)
        except:
            tab_gain.append(np.nan)


    return tab_gain


def nb_saccade(df):
    list_saccade_index = df.NSLR_Class[df.NSLR_Class == 'Saccade'].index.values
    group_saccade = np.split(list_saccade_index, np.where(np.diff(list_saccade_index) != 1)[0] + 1)
    nb_saccade = len([group[0] for group in group_saccade if df.Valid_Saccade.loc[group].all() == True])

    return nb_saccade


def peak_velocity_saccades(df, name_task):
    tab_df = get_saccade(df)

    tab_peak_velocity = []
    for df_saccade in tab_df:
        try:
            eye_velocity = np.diff(df_saccade[CONST_PARAMETERS[name_task][0][0]]) / np.diff(df_saccade.timestamp)
            tab_peak_velocity.append(np.max(np.abs(eye_velocity)))
        except:
            tab_peak_velocity.append(np.nan)

    return tab_peak_velocity


def gain_saccades(df, df_ball_position, name_task):
    tab_df = get_saccade(df)
    tab_gain_saccade = []
    ball_amplitude = np.abs(np.max(df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])])- np.min(
        df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])]))

    for df_saccade in tab_df:
        saccade_amplitude = np.abs(df_saccade[CONST_PARAMETERS[name_task][0][0]].iloc[-1] - df_saccade[CONST_PARAMETERS[name_task][0][0]].iloc[0])
        tab_gain_saccade.append(saccade_amplitude/ball_amplitude)

    return tab_gain_saccade


def phase_lag(df, df_ball_position, name_task):
    phase_lag = []

    #crop df to df_ball_position timestamp
    df_crop = df[df.timestamp.between(df_ball_position.timestamp.iloc[0], df_ball_position.timestamp.iloc[-1])]

    #Interpolation of the ball_position to match the pupil data
    f = interp1d(df_ball_position.timestamp, df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])])
    position_interpolated = f(df_crop.timestamp)

    phase_lag = df_crop[CONST_PARAMETERS[name_task][0][0]].values - position_interpolated

    return np.median(phase_lag)


def gain_pursuit(df, df_ball_position, name_task):
    # Find indices of consecutive timestamps with the same value
    same_timestamps = df.iloc[np.where(np.diff(df.timestamp) == 0)[0]].index

    # Remove rows with same timestamp from the DataFrame
    if len(same_timestamps) > 0:
        df = df.drop(same_timestamps + 1)

    ball_velocity = np.diff(df_ball_position[utils.eq_axis(CONST_PARAMETERS[name_task][0][0])]) / np.diff(
        df_ball_position.timestamp)
    eye_velocity = np.diff(df[CONST_PARAMETERS[name_task][0][0]]) / np.diff(df.timestamp)

    # Replace infinite values with NaN
    eye_velocity[np.isinf(eye_velocity)] = np.nan

    return np.nanmean(eye_velocity) / np.mean(ball_velocity)


def gaze_velocity(df, name_task):

    eye_velocity = np.diff(df[CONST_PARAMETERS[name_task][0][0]]) / np.diff(df.timestamp)

    return np.median(eye_velocity)


def main(df: pd.DataFrame, task_timestamp, df_ball_position: pd.DataFrame, path_saccade_analysis: str, name_task: str):
    # TODO: complete check_saccade function
    df = check_saccade_consistency(df=df, df_ball_position=df_ball_position,
                                   path_saccade_analysis=path_saccade_analysis, name_task=name_task)

    if name_task == 'biflicker':
        df = check_anticipated_saccade(df, task_timestamp)
        result = {}
        result['Directional_error'], df = directional_error(df, task_timestamp, df_ball_position, name_task)
        result['Latency_of_the_first_saccade_ms'] = latency(df, task_timestamp, df_ball_position, name_task)
        result['Peak_velocity_of_the_first_saccade'] = peak_velocity(df, task_timestamp, df_ball_position, name_task)
        result['Gain_of_the_first_saccade'] = gain(df, task_timestamp, df_ball_position, name_task)
        result['Saccade_count'] = nb_saccade(df)

    elif name_task == 'vertical_biflicker':
        df = check_anticipated_saccade(df, task_timestamp)
        result = {}
        result['Directional_error'], df = directional_error(df, task_timestamp, df_ball_position, name_task)
        result['Latency_of_the_first_saccade_ms'] = latency(df, task_timestamp, df_ball_position, name_task)
        result['Peak_velocity_of_the_first_saccade'] = peak_velocity(df, task_timestamp, df_ball_position, name_task)
        result['Gain_of_the_first_saccade'] = gain(df, task_timestamp, df_ball_position, name_task)
        result['Saccade_count'] = nb_saccade(df)


    elif name_task == 'antibiflicker':
        df = check_anticipated_saccade(df, task_timestamp)
        result = {}
        result['Directional_error'], df = directional_error(df, task_timestamp, df_ball_position, name_task)
        result['Latency_of_the_first_saccade_ms'] = latency(df, task_timestamp, df_ball_position, name_task)
        result['Peak_velocity_of_the_first_saccade'] = peak_velocity(df, task_timestamp, df_ball_position, name_task)
        result['Gain_of_the_first_saccade'] = gain(df, task_timestamp, df_ball_position, name_task)
        result['Saccade_count'] = nb_saccade(df)

    elif name_task == 'vertical_antibiflicker':
        df = check_anticipated_saccade(df, task_timestamp)
        result = {}
        result['Directional_error'], df = directional_error(df, task_timestamp, df_ball_position, name_task)
        result['Latency_of_the_first_saccade_ms'] = latency(df, task_timestamp, df_ball_position, name_task)
        result['Peak_velocity_of_the_first_saccade'] = peak_velocity(df, task_timestamp, df_ball_position, name_task)
        result['Gain_of_the_first_saccade'] = gain(df, task_timestamp, df_ball_position, name_task)
        result['Saccade_count'] = nb_saccade(df)

    elif name_task == 'velo_horizontal':
        result = {}
        result['Peak_velocity_saccades'] = peak_velocity_saccades(df, name_task)
        result['Gain_saccades'] = gain_saccades(df, df_ball_position, name_task)
        result['Saccade_count'] = nb_saccade(df)


    elif name_task == 'velo_vertical':
        result = {}
        result['Peak_velocity_saccades'] = peak_velocity_saccades(df, name_task)
        result['Gain_saccades'] = gain_saccades(df, df_ball_position, name_task)
        result['Saccade_count'] = nb_saccade(df)

    elif name_task == 'circle':
        result = {}
        result['Phase_lag'] = phase_lag(df, df_ball_position, name_task)
        result['Gain_pursuit'] = gain_pursuit(df, df_ball_position, name_task)
        result['Gaze_velocity'] = gaze_velocity(df, name_task)
        result['Saccade_count'] = nb_saccade(df)

    elif name_task == 'sine':
        result = {}
        result['Phase_lag'] = phase_lag(df, df_ball_position, name_task)
        result['Gain_pursuit'] = gain_pursuit(df, df_ball_position, name_task)
        result['Gaze_velocity'] = gaze_velocity(df, name_task)
        result['Saccade_count'] = nb_saccade(df)

    else:
        result = {}

    return result
