from . import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    df_constants_experiment = pd.read_csv(
        os.path.join(PATH_PROJECT, 'resources', "experiment_constant_straight.csv"))
    df_constants_reliability = pd.read_csv(
        os.path.join(PATH_PROJECT, 'resources', "reliability_constant_straight.csv"))

    fig, ax = plt.subplots(2, len(df_constants_reliability.keys()), figsize=(15, 8))

    for i in range(len(df_constants_reliability.keys())):
        ax[0, i].hist(df_constants_reliability.iloc[:, i])
        ax[0, i].set_title('Reliability: ' + str(df_constants_reliability.keys()[i]))
        ax[0, i].axvline(np.median(df_constants_reliability.iloc[:, i]), c='r', label='Median')
        ax[0, i].axvline(np.mean(df_constants_reliability.iloc[:, i]), c='y', label='Mean')

    for i in range(len(df_constants_experiment.keys())):
        ax[1, i].hist(df_constants_experiment.iloc[:, i])
        ax[1, i].set_title('Exprimentation: ' + str(df_constants_experiment.keys()[i]))
        ax[1, i].axvline(np.median(df_constants_reliability.iloc[:, i]), c='r', label='Median')
        ax[1, i].axvline(np.mean(df_constants_reliability.iloc[:, i]), c='y', label='Mean')
        ax[1, i].set_xlabel('World coord (vect)')

    # Label

    ax[0, 0].set_ylabel('Frequency')
    ax[1, 0].set_ylabel('Frequency')

    fig.suptitle('X, Y, Z Straight constants histogramm')
    plt.legend()
    fig.savefig(os.path.join(PATH_PROJECT, 'result_analysis', 'x_y_z_straight.png'))
