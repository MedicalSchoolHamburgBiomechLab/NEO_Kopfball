# Constants used in the NEO-Kopfball project
import sys
import os

# PROJECT PATH
PATH_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONST_DATA = 'old'
CONST_DATA_TYPE = 'reliability'  # experiment or reliability
CONST_STRAIGHT = 'ID_Session_NEO_Kopfball_reliability_AB_update.xlsx'  # ID Mapping table link to data: ID_sessions_Hauptstudie (experimentation) or ID_Session_NEO_Kopfball_reliability_AB_update (reliability)

# Define constant name to find the table related to it in the eyetrax file
CONST_LEFT_EYE_2D = 'LeftEye_'
CONST_LEFT_EYE_3D = 'LeftEye3d'
CONST_EVENTS = 'Events_'
CONST_BALL = 'BallPositions_'
CONST_PARAMETERS = {

    "biflicker": [['phi', 'theta'], 'degree',
                  ['Directional_error', 'Latency_of_the_first_saccade_ms', 'Peak_velocity_of_the_first_saccade',
                   'Gain_of_the_first_saccade', 'Saccade_count']],

    "vertical_biflicker": [['theta', 'phi'], 'degree', ['Directional_error', 'Latency_of_the_first_saccade_ms',
                                                        'Peak_velocity_of_the_first_saccade',
                                                        'Gain_of_the_first_saccade', 'Saccade_count']],

    # "vertical_antibiflicker": [['theta', 'phi'], 'degree', ['Directional_error', 'Latency_of_the_first_saccade_ms',
    #                                                         'Peak_velocity_of_the_first_saccade',
    #                                                         'Gain_of_the_first_saccade', 'Saccade_count']],

    "antibiflicker": [['phi', 'theta'], 'degree',
                      ['Directional_error', 'Latency_of_the_first_saccade_ms', 'Peak_velocity_of_the_first_saccade',
                       'Gain_of_the_first_saccade', 'Saccade_count']],

    "velo_horizontal": [['phi', 'theta'], 'degree',
                        ['Peak_velocity_saccades', 'Gain_saccades', 'Saccade_count']],

    "velo_vertical": [['theta', 'phi'], 'degree',
                      ['Peak_velocity_saccades', 'Gain_saccades', 'Saccade_count']],

    "circle": [['phi', 'theta'], 'degree', ['Phase_lag', 'Gain_pursuit', 'Gaze_velocity', 'Saccade_count']],

    "sine": [['theta', 'phi'], 'degree', ['Phase_lag', 'Gain_pursuit', 'Gaze_velocity', 'Saccade_count']]
}
CONST_PLOT_WINDOW = 0  # second added as extra range for the task plot
CONST_COLOR = {
    "Fixation": 'red',
    "Saccade": 'purple',
    "Smooth Pursuit": 'yellow'
}
CONST_SACCADE = {
    'min_timestamp': 0.01,
    'max_timestamp': 0.7,
    'window_first_saccade': 0.1
}

# MULTIPROCESSING CONSTANTS
gettrace = getattr(sys, 'gettrace', None)

if gettrace():
    DEBUG_MODE = True
else:
    DEBUG_MODE = False

