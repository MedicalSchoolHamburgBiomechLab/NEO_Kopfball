import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cateyes
import utils as utils
from src.constants import *
from src.constants.calculated_constants import *
import warnings
import pandas as pd
from src.processing import outcomes_processing

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# Create global plot
figure_global, axis_global = plt.subplots(2, 1)
figure_global.set_figheight(15)
figure_global.set_figwidth(20)
global_subject_plot = (figure_global, axis_global)
str_error = ""

def load_events(path: str):
    df_events = utils.sqlite3_to_df(path_file=path, name_table=CONST_EVENTS)
    for i in range(len(df_events.iloc[:, 2])):
        df_events.iloc[i, 1] = df_events.iloc[i, 1].replace(" ", "_")
        if df_events.iloc[i, 2] == "instruction":
            df_events.iloc[i, 1] = "instruction"

    df_events = df_events.sort_values(by='timestamp')
    df_events.reset_index(drop=True)
    return df_events


def load_left_value(path: str, name_table=CONST_LEFT_EYE_2D):
    df_left = utils.sqlite3_to_df(path_file=path, name_table=name_table)
    df_left = df_left.sort_values(by='timestamp')
    df_left.reset_index(drop=True)
    return df_left


def load_ball_position(path: str):
    df_ball_positions = utils.sqlite3_to_df(path_file=path, name_table=CONST_BALL)
    df_ball_positions = df_ball_positions.sort_values(by='timestamp')
    df_ball_positions.reset_index(drop=True)

    return df_ball_positions


def create_task_folder(path: str, list_task):
    for task in list_task:
        task = task.replace(" ", "_")
        if not os.path.isdir(os.path.join(path, task)):
            os.mkdir(os.path.join(path, task))


def rotation(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    """
    # unit vectors
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)
    # dimension of the space and identity
    dim = u.size
    I = np.identity(dim)
    # the cos angle between the vectors
    c = np.dot(u, Ru)
    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        return I
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        # the cross product matrix of a vector to rotate around
        K = np.outer(Ru, u) - np.outer(u, Ru)
        # Rodrigues' formula
        return I + K + (K @ K) / (1 + c)


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

    global str_error

    if (CONST_X_Q1 < x_straight < CONST_X_Q3) and (CONST_Y_Q1 < y_straight < CONST_Y_Q3) and (
            CONST_Z_Q1 < z_straight < CONST_Z_Q3):
        str_error = ""
        return x_straight, y_straight, z_straight

    else:

        str_error = "rotation median value invalid : x=" + str(x_straight) + "  y=" + str(y_straight) + "  z=" + str(
            z_straight)
        raise Exception("rotation median value invalid")


def rotate_data(df, x_traight, y_straight, z_straight):
    straight_vector = np.array([x_traight, y_straight, z_straight])

    # r = R.from_euler('z', straight_vector)
    r = rotation(straight_vector, np.array([0, 0, 1]))

    df["phi_origin"] = df["phi"]
    df["theta_origin"] = df["theta"]

    rotated_vec = []

    for index, row in df.iterrows():
        left_pupil_vec = np.array([row['c_normal_x'], row['c_normal_y'], row['c_normal_z']])
        # rotated_vec.append(r @ left_pupil_vec) -> other way to apply the dot multiplication
        rotated_vec.append(np.dot(r, left_pupil_vec))

    rotated_vec = np.transpose(rotated_vec)

    df["phi"] = np.arctan2(rotated_vec[0, :], rotated_vec[2, :]) * 180 / np.pi
    df["theta"] = -np.arctan2(rotated_vec[1, :], rotated_vec[2, :]) * 180 / np.pi


    return df


def data_formatting(df_values, df_events, df_ball_position):
    # Set timestamp to time records
    CONST_TIMESTAMP = np.min(df_values['timestamp'])

    df_values['timestamp_origin'] = df_values['timestamp']
    df_values['timestamp'] = df_values['timestamp_origin'] - CONST_TIMESTAMP
    df_events['timestamp_origin'] = df_events['timestamp']
    df_events['timestamp'] = df_events['timestamp_origin'] - CONST_TIMESTAMP
    df_ball_position['timestamp_origin'] = df_ball_position['timestamp']
    df_ball_position['timestamp'] = df_ball_position['timestamp_origin'] - CONST_TIMESTAMP

    # Eye angles calculation
    df_values.x = np.degrees(df_values.x)
    df_values.y = np.degrees(df_values.y)
    df_values.phi = np.degrees(df_values.phi)
    df_values.theta = np.degrees(df_values.theta)

    # Ball angles calculation part
    df_ball_position['x_angles'] = np.degrees(np.arctan((df_ball_position.x / df_ball_position.z)))
    df_ball_position['x_angles'] = df_ball_position['x_angles'].fillna(0)
    df_ball_position['y_angles'] = np.degrees(np.arctan((df_ball_position.y / df_ball_position.z)))
    df_ball_position['y_angles'] = df_ball_position['y_angles'].fillna(0)

    x_straight, y_straight, z_straight = x_y_z_straight(df_values, df_ball_position)

    df_values = rotate_data(df_values, x_straight, y_straight, z_straight)

    return df_values, df_events, df_ball_position


def task_processing(path: str, name_task: str, df_events, df_values, df_ball_position, **kwargs):
    global_plot = kwargs.get('global_plot', False)
    save_data_processed = kwargs.get('save_data_processed', True)
    save_plot = kwargs.get('save_plot', True)
    outcomes_calculation = kwargs.get('outcomes_calculation', True)
    saccade_detection = kwargs.get('saccade_detection', True)

    task_timestamp = df_events.timestamp[df_events.event == name_task].values


    # Add +1 timestamp value to have the entiere number of task
    indice = np.where(df_events.timestamp == task_timestamp[-1])
    if not indice[0][-1] == len(df_events) - 1:
        task_timestamp = np.append(task_timestamp, df_events.timestamp[indice[0][-1] + 1])

    if (task_timestamp[0] - CONST_PLOT_WINDOW > np.min(df_values.timestamp)) and (
            task_timestamp[-1] + CONST_PLOT_WINDOW < np.max(df_values.timestamp)):
        task_timestamp_plot = [task_timestamp[0] - CONST_PLOT_WINDOW, task_timestamp[-1] + CONST_PLOT_WINDOW]
    else:
        task_timestamp_plot = [task_timestamp[0], task_timestamp[-1]]

    df_task = df_values[
        (df_values.timestamp >= task_timestamp_plot[0]) & (df_values.timestamp <= task_timestamp_plot[1])]

    df_task_ball = df_ball_position[
        (df_ball_position.timestamp >= task_timestamp_plot[0]) & (
                df_ball_position.timestamp <= task_timestamp_plot[1])]

    # FIXATION/SACCADE DETECTION PART
    if not saccade_detection:
        if os.path.isfile(os.path.join(os.path.dirname(path), name_task, f'df_{name_task}.pickle')):
            df_task = pd.read_pickle(os.path.join(os.path.dirname(path), name_task, f'df_{name_task}.pickle'))
        else:
            df_task = fixation_saccade_detection(df_task)
    else:
        df_task = fixation_saccade_detection(df_task)

    # Check that the df got saccade/fixation label
    try:
        df_task.NSLR_Class.iloc[0]
    except:
        df_task = fixation_saccade_detection(df_task)

    # PARAMETERS CALCULATION
    if outcomes_calculation:

        outcomes_result = outcomes_processing.main(df=df_task, task_timestamp=task_timestamp,
                                                   df_ball_position=df_task_ball, path_saccade_analysis=path,
                                                   name_task=name_task)
        # SAVE RESULT IN CSV FILE
        with open(os.path.join(os.path.dirname(path), name_task, name_task + "_outcomes.csv"),
                  'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['SACCADE_PARAMETERS', 'PARAMETERS_VALUES'])
            writer.writerow(['Task_Name', name_task])
            for key, value in outcomes_result.items():
                writer.writerow([key, value])

    # SAVE PLOT PART
    if save_plot:
        save_task_plot(path=path, name_task=[name_task, 0], df=df_task, df_ball=df_task_ball,
                       tab_task_timestamp=task_timestamp)

    # SAVE DF IN PICKLE FILE
    if save_data_processed:
        df_task.to_pickle(os.path.join(os.path.dirname(path), name_task, "df_" + name_task + ".pickle"))

    if global_plot:
        save_crop_plot(name_task=name_task, df=df_task, tab_task_timestamp=task_timestamp,
                       outcomes_result=outcomes_result, path=path)
        set_global_plot(name_task=name_task, df=df_task, tab_task_timestamp=task_timestamp,
                        outcomes_result=outcomes_result)


def fixation_saccade_detection(df):
    segments_nslr, classes_nslr = cateyes.classify_nslr_hmm(df["x"], df["y"], df["timestamp"], optimize_noise=False)
    df["NSLR_Class"] = classes_nslr
    df["NSLR_Segment"] = segments_nslr

    return df


def save_crop_plot(name_task: str, df, tab_task_timestamp, outcomes_result, path):
    figure_subject, axis_subject = plt.subplots(2, 1)
    figure_subject.set_figheight(15)
    figure_subject.set_figwidth(20)

    segments_events = cateyes.continuous_to_discrete(df["timestamp"], df["NSLR_Segment"], df["NSLR_Class"])
    segments_events[0].append(np.max(df.timestamp))
    segments_events[1].append(df.NSLR_Class.iloc[-1])

    for i in range(1, len(tab_task_timestamp) - 1):

        x_value = df['timestamp'][df['timestamp'].between(tab_task_timestamp[i - 1], tab_task_timestamp[i + 1],
                                                          inclusive='both')].values - tab_task_timestamp[i]
        y_value_0 = df[CONST_PARAMETERS[name_task][0][0]][
            df['timestamp'].between(tab_task_timestamp[i - 1], tab_task_timestamp[i + 1],
                                    inclusive='both')].values
        y_value_1 = df[CONST_PARAMETERS[name_task][0][1]][
            df['timestamp'].between(tab_task_timestamp[i - 1], tab_task_timestamp[i + 1],
                                    inclusive='both')].values

        axis_subject[0].plot(x_value, y_value_0, alpha=0.4, c='b')
        axis_subject[1].plot(x_value, y_value_1, alpha=0.4, c='b')

        # Plot task_timestamp
        axis_subject[0].axvline(x=0, color='r', linestyle="--")
        axis_subject[1].axvline(x=0, color='r', linestyle="--")

        try:
            # Plot latency
            axis_subject[0].axvline(x=(outcomes_result['Latency_of_the_first_saccade_ms'][i - 1] / 1000), color='grey',
                                    linestyle="--")
            axis_subject[1].axvline(x=(outcomes_result['Latency_of_the_first_saccade_ms'][i - 1] / 1000), color='grey',
                                    linestyle="--")
        except:
            pass

        axis_subject[0].set_title(f"{name_task} global plot")
        axis_subject[0].set_ylabel(f"{CONST_PARAMETERS[name_task][0][0]} ({CONST_PARAMETERS[name_task][1]})")
        axis_subject[0].set_xlabel(f"timestamp (s)")

        axis_subject[1].set_ylabel(f"{CONST_PARAMETERS[name_task][0][1]} ({CONST_PARAMETERS[name_task][1]})")
        axis_subject[1].set_xlabel(f"timestamp (s)")

        grey_patch = mpatches.Patch(color='grey', label='Latency')
        red_patch = mpatches.Patch(color='red', label='Stimulus timestamp')
        blue_patch = mpatches.Patch(color='blue', label='eye angle')
        axis_subject[0].legend(handles=[red_patch, grey_patch, blue_patch], loc='lower right')
        axis_subject[1].legend(handles=[red_patch, grey_patch, blue_patch], loc='lower right')

        # Save figure in the corresponding folder task
        figure_subject.savefig(os.path.join(os.path.dirname(path), name_task, f"{name_task}_crop.png"))
        plt.close(figure_subject)


def set_global_plot(name_task: str, df, tab_task_timestamp, outcomes_result):
    global global_subject_plot
    axis_global = global_subject_plot[1]

    segments_events = cateyes.continuous_to_discrete(df["timestamp"], df["NSLR_Segment"], df["NSLR_Class"])
    segments_events[0].append(np.max(df.timestamp))
    segments_events[1].append(df.NSLR_Class.iloc[-1])

    for i in range(1, len(tab_task_timestamp) - 1):

        x_value = df['timestamp'][df['timestamp'].between(tab_task_timestamp[i - 1], tab_task_timestamp[i + 1],
                                                          inclusive='both')].values - tab_task_timestamp[i]
        y_value_0 = df[CONST_PARAMETERS[name_task][0][0]][
            df['timestamp'].between(tab_task_timestamp[i - 1], tab_task_timestamp[i + 1],
                                    inclusive='both')].values
        y_value_1 = df[CONST_PARAMETERS[name_task][0][1]][
            df['timestamp'].between(tab_task_timestamp[i - 1], tab_task_timestamp[i + 1],
                                    inclusive='both')].values

        axis_global[0].plot(x_value, y_value_0, alpha=0.4, c='b')
        axis_global[1].plot(x_value, y_value_1, alpha=0.4, c='b')

        # Plot task_timestamp
        axis_global[0].axvline(x=0, color='r', linestyle="--")
        axis_global[1].axvline(x=0, color='r', linestyle="--")

        try:
            # Plot latency
            axis_global[0].axvline(x=(outcomes_result['Latency_of_the_first_saccade_ms'][i - 1] / 1000), color='grey',
                                   linestyle="--")
            axis_global[1].axvline(x=(outcomes_result['Latency_of_the_first_saccade_ms'][i - 1] / 1000), color='grey',
                                   linestyle="--")
        except:
            pass


def save_global_plot(path, name_task: str):
    global global_subject_plot
    figure_global = global_subject_plot[0]
    axis_global = global_subject_plot[1]

    axis_global[0].set_title(f"{name_task} global plot")
    axis_global[0].set_ylabel(f"{CONST_PARAMETERS[name_task][0][0]} ({CONST_PARAMETERS[name_task][1]})")
    axis_global[0].set_xlabel(f"timestamp (s)")

    axis_global[1].set_ylabel(f"{CONST_PARAMETERS[name_task][0][1]} ({CONST_PARAMETERS[name_task][1]})")
    axis_global[1].set_xlabel(f"timestamp (s)")

    grey_patch = mpatches.Patch(color='grey', label='Latency')
    red_patch = mpatches.Patch(color='red', label='Stimulus timestamp')
    blue_patch = mpatches.Patch(color='blue', label='eye angle')
    axis_global[0].legend(handles=[red_patch, grey_patch, blue_patch], loc='lower right')
    axis_global[1].legend(handles=[red_patch, grey_patch, blue_patch], loc='lower right')

    # Save figure in the corresponding folder task
    # figure.savefig(os.path.join(PATH_PROJECT, 'result_analysis', f"{name_task}.png"))
    figure_global.savefig(os.path.join(path, f"{name_task}_crop.png"))
    plt.close(figure_global)


def save_task_plot(path: str, name_task: str, df, df_ball, tab_task_timestamp):
    figure_, axis_ = plt.subplots(2, 1)
    figure_.set_figheight(15)
    figure_.set_figwidth(20)

    segments_events = cateyes.continuous_to_discrete(df["timestamp"], df["NSLR_Segment"], df["NSLR_Class"])
    segments_events[0].append(np.max(df.timestamp))
    segments_events[1].append(df.NSLR_Class.iloc[-1])

    for i in range(len(segments_events[0]) - 1):

        try:
            df_segment = df[df['timestamp'].between(segments_events[0][i], segments_events[0][i + 1], inclusive='both')]
            axis_[0].plot(df_segment.timestamp, df_segment[CONST_PARAMETERS[name_task[0]][0][0]],
                          color=CONST_COLOR[segments_events[1][i]], label=segments_events[1][i])
            axis_[1].plot(df_segment.timestamp, df_segment[CONST_PARAMETERS[name_task[0]][0][1]],
                          color=CONST_COLOR[segments_events[1][i]], label=segments_events[1][i])
        except:
            df_segment = df[df['timestamp'].between(segments_events[0][i], segments_events[0][i + 1], inclusive='both')]
            axis_[0].plot(df_segment.timestamp, df_segment[CONST_PARAMETERS[name_task[0]][0][0]], color='black',
                          label='Others')
            axis_[1].plot(df_segment.timestamp, df_segment[CONST_PARAMETERS[name_task[0]][0][1]], color='black',
                          label='Others')

    axis_[0].plot(df_ball['timestamp'], df_ball[utils.eq_axis(CONST_PARAMETERS[name_task[0]][0][0])],
                  label='Ball x angles', c='b')
    axis_[1].plot(df_ball['timestamp'], df_ball[utils.eq_axis(CONST_PARAMETERS[name_task[0]][0][1])],
                  label='Ball y angles', c='b')

    # Add start/stop stimulus
    for value in tab_task_timestamp:
        axis_[0].axvline(x=value, color='r', linestyle="--")
        axis_[1].axvline(x=value, color='r', linestyle="--")

    axis_[0].set_title(f"{name_task[0]} n°{name_task[1] + 1}")
    axis_[0].set_ylabel(f"{CONST_PARAMETERS[name_task[0]][0][0]} ({CONST_PARAMETERS[name_task[0]][1]})")
    axis_[0].set_xlabel(f"timestamp (s)")

    axis_[1].set_ylabel(f"{CONST_PARAMETERS[name_task[0]][0][1]} ({CONST_PARAMETERS[name_task[0]][1]})")
    axis_[1].set_xlabel(f"timestamp (s)")

    red_patch = mpatches.Patch(color='red', label='Fixation')
    purple_patch = mpatches.Patch(color='purple', label='Saccade')
    yellow_patch = mpatches.Patch(color='yellow', label='Smooth Pursuit')
    blue_x_patch = mpatches.Patch(color='blue', label='Ball x angles')
    blue_y_patch = mpatches.Patch(color='blue', label='Ball y angles')
    axis_[0].legend(handles=[red_patch, purple_patch, yellow_patch, blue_x_patch], loc='lower right')
    axis_[1].legend(handles=[red_patch, purple_patch, yellow_patch, blue_y_patch], loc='lower right')

    # Save figure in the corresponding folder task
    figure_.savefig(os.path.join(os.path.dirname(path), name_task[0], f"{name_task[0]}_{name_task[1] + 1}.png"))
    plt.close(figure_)


def main(**kwargs):

    task_params = {k: kwargs[k] for k in
                   ('save_plot', 'outcomes_calculation', 'saccade_detection', 'save_data_processed', 'global_plot') if
                   k in kwargs}
    list_files = kwargs.get('path_subject',
                            utils.get_files(os.path.join(PATH_PROJECT, 'data', 'new'), extension=".eyetrax"))

    if not os.path.isdir(os.path.join(PATH_PROJECT, 'result_analysis')):
        os.mkdir(os.path.join(PATH_PROJECT, 'result_analysis'))

    for path_subject in list_files:
        print(path_subject)

        # if path_subject in PATH_UNUSABLE_FILE:
        #     print(f'not usable eyetrack file')
        # else:
        try:
            # Load data
            df_left = load_left_value(path=path_subject, name_table=CONST_LEFT_EYE_3D)
            df_events = load_events(path=path_subject)
            df_ball_position = load_ball_position(path=path_subject)
            # list_events = kwargs.get('name_task', np.delete(np.unique(df_events.event),
            #                                                 np.where(np.unique(df_events.event) == 'instruction')))
            list_events = kwargs.get('name_task', list(CONST_PARAMETERS.keys()))

            # Create Folder
            create_task_folder(os.path.dirname(path_subject), list_events)

            # Data processing
            df_left, df_events, df_ball_position = data_formatting(df_left,
                                                                   df_events,
                                                                   df_ball_position)

            for event in list_events:
                task_processing(path=path_subject, name_task=event, df_events=df_events, df_values=df_left,
                                df_ball_position=df_ball_position, **task_params)

            save_global_plot(path=os.path.join(PATH_PROJECT, 'result_analysis'), name_task=list_events[0])
        except Exception as exception:
            print(exception)

    plt.close()

    if len(str_error) != 0:
        return [path_subject, str_error]
    else:
        return []
