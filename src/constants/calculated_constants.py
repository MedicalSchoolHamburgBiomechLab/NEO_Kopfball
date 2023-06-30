import numpy as np
import pandas as pd
from .__init__ import *

# PATH ressources
try:
    PATH_UNUSABLE_FILE = pd.read_excel(os.path.join(PATH_PROJECT, 'resources', 'Path_unusuable_eyetrack_files.xlsx'))
    PATH_UNUSABLE_FILE = PATH_UNUSABLE_FILE.Path_unusuable_eyetrack_file.values
except Exception as e:
    print(e)
    PATH_UNUSABLE_FILE = []

# Data processing constant
try:
    df_x_y_z_straight = pd.read_csv(os.path.join(PATH_PROJECT, 'resources',
                                                 CONST_DATA_TYPE + '_constant_straight.csv'))  # Switch value regarding dataset used -> experiment_... OR reliability_...

    CONST_X_Q1 = np.percentile(df_x_y_z_straight.X_Straight.values, 5)
    CONST_X_Q3 = np.percentile(df_x_y_z_straight.X_Straight.values, 95)
    CONST_Y_Q1 = np.percentile(df_x_y_z_straight.Y_Straight.values, 5)
    CONST_Y_Q3 = np.percentile(df_x_y_z_straight.Y_Straight.values, 95)
    CONST_Z_Q1 = np.percentile(df_x_y_z_straight.Z_Straight.values, 5)
    CONST_Z_Q3 = np.percentile(df_x_y_z_straight.Z_Straight.values, 95)

    print('Constants process ✅')

except:
    raise Exception(CONST_DATA_TYPE + "_straight_constant.csv file NOT FOUND")

try:
    DF_ID_SESSION = pd.read_excel(
        os.path.join(PATH_PROJECT, 'resources', CONST_STRAIGHT))
except:
    raise Exception('ID_Session table NOT FOUND')
