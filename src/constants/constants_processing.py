import numpy as np
from . import *
import utils
from concurrent.futures import ProcessPoolExecutor
import queue
import pandas as pd

# Init queue
q_x = queue.Queue()
q_y = queue.Queue()
q_z = queue.Queue()


def load_left_value(path: str, name_table=CONST_LEFT_EYE_2D):
    df_left = utils.sqlite3_to_df(path_file=path, name_table=name_table)
    df_left = df_left.sort_values(by='timestamp')
    df_left.reset_index(drop=True)
    return df_left


def load_ball_position(path: str):
    df_ball_positions = utils.sqlite3_to_df(path_file=path, name_table=CONST_BALL)
    df_ball_positions = df_ball_positions.sort_values('timestamp')
    df_ball_positions.reset_index(drop=True)

    return df_ball_positions


def add_to_queue(result):
    if not np.isnan(result[0]):
        q_x.put(result[0])
    if not np.isnan(result[1]):
        q_y.put(result[1])
    if not np.isnan(result[2]):
        q_z.put(result[2])



def x_y_z_straight(df, df_ball_position):
    """
    function that detect when the subject looks straight in front of him and return median of phi/theta angles
    :param df: eyetrack file of the left eye
    :param df_ball_position: dataframe of the ball position
    :return:phi/theta angles in degrees
    """
    segment_zero = df_ball_position[
        (df_ball_position['x_angles'] == 0) & (df_ball_position['y_angles'] == 0)].index.values
    group_segment_zero = np.split(segment_zero, np.where(np.diff(segment_zero) != 1)[0] + 1)

    # --keep only segment > 1s
    tab_segment = []
    for segment in group_segment_zero:
        if df_ball_position['timestamp'][segment[-1]] - df_ball_position['timestamp'][segment[0]] > 1:
            tab_segment.append(segment)

    x_normal_eye_value = []
    y_normal_eye_value = []
    z_normal_eye_value = []

    for tab in tab_segment:
        timestamp_eye = df['timestamp'].between(df_ball_position['timestamp'][tab[0]],
                                                df_ball_position['timestamp'][tab[-1]], inclusive='both')
        x_normal_eye_value.append(np.median(df['c_normal_x'][timestamp_eye].values))
        y_normal_eye_value.append(np.median(df['c_normal_y'][timestamp_eye].values))
        z_normal_eye_value.append(np.median(df['c_normal_z'][timestamp_eye].values))

    x_straight = np.median(x_normal_eye_value)
    y_straight = np.median(y_normal_eye_value)
    z_straight = np.median(z_normal_eye_value)

    return x_straight, y_straight, z_straight


def get_constants(path_subject):
    try:
        df_values = load_left_value(path=path_subject, name_table=CONST_LEFT_EYE_3D)
        df_ball_position = load_ball_position(path=path_subject)

        # Set timestamp to time records
        CONST_TIMESTAMP = np.min(df_values['timestamp'])

        df_values['timestamp_origin'] = df_values['timestamp']
        df_values['timestamp'] = df_values['timestamp_origin'] - CONST_TIMESTAMP
        df_ball_position['timestamp_origin'] = df_ball_position['timestamp']
        df_ball_position['timestamp'] = df_ball_position['timestamp_origin'] - CONST_TIMESTAMP

        # Ball angles calculation part
        df_ball_position['x_angles'] = np.degrees(np.arctan((df_ball_position.x / df_ball_position.z)))
        df_ball_position['x_angles'] = df_ball_position['x_angles'].fillna(0)
        df_ball_position['y_angles'] = np.degrees(np.arctan((df_ball_position.y / df_ball_position.z)))
        df_ball_position['y_angles'] = df_ball_position['y_angles'].fillna(0)

        return x_y_z_straight(df_values, df_ball_position)
    except Exception as e:
        print(e)
        return np.nan, np.nan, np.nan


def main():
    list_path = utils.get_files(os.path.join(PATH_PROJECT, 'data', CONST_DATA), extension=".eyetrax")
    name_constants_file = CONST_DATA_TYPE + "_constant_straight"

    tab_x_straight = []
    tab_y_straight = []
    tab_z_straight = []

    # OPTIMIZE PROCESS
    with ProcessPoolExecutor(utils.get_cpu_core_count()) as executor:
        for path in list_path:
            future = executor.submit(get_constants, path_subject=path)

            future.add_done_callback(lambda x: add_to_queue(x.result()))

    for item in q_x.queue:
        if type(item) == int or type(item) == float or type(item) == np.float64:
            tab_x_straight.append(item)

    for item in q_y.queue:
        if type(item) == int or type(item) == float or type(item) == np.float64:
            tab_y_straight.append(item)

    for item in q_z.queue:
        if type(item) == int or type(item) == float or type(item) == np.float64:
            tab_z_straight.append(item)

    df = pd.DataFrame()

    df['X_Straight'] = tab_x_straight
    df['Y_Straight'] = tab_y_straight
    df['Z_Straight'] = tab_z_straight

    df.to_csv(os.path.join(PATH_PROJECT, 'resources', f"{name_constants_file}.csv"), index=False)
