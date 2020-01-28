"""
Resample irregular time series to regular intervals

Arguments:
    file -- csv file containing input data
    output_file -- csv file with resampled output
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

COLUMNS = ['time', 'value']
COLUMN_TYPES = {'time': 'float', 'value': 'float'}
ROWS_TO_SKIP = 3
DECIMAL_POINTS = 4

pd.options.mode.chained_assignment = None  # default='warn'


def resample_dataframe(dataframe, max_value, step, method, order=1):
    """
    Resample input dataframe to regular intervals with fixed [step]

    To prevent imprecise float comparisons, index is converted to integers
    by multiplying it to 10^DECIMAL_POINTS first, and divided back after interpolation.
    """
    precision = 10**DECIMAL_POINTS

    df = dataframe.copy()
    df.drop_duplicates('time', keep='last', inplace=True)  # prevent duplicate indices
    df['time'] = df['time'].mul(precision).astype(int)
    df.set_index('time', inplace=True)

    regular_ticks = pd.Index(np.arange(0, max_value*precision, step*precision))
    new_index = df.index.union(regular_ticks).drop_duplicates(keep='last')
    df = df.reindex(new_index)

    df = df.interpolate(method=method, order=order)  # fill inner NaNs by interpolation
    df = df.ffill().bfill()  # fill NaNs at the ends of dataframe by extending nearest values over them
    df = df.round(DECIMAL_POINTS)

    filtered_dataframe = df[df.index % int(step*precision) == 0]  # keep only regular intervals
    filtered_dataframe.index.name = 'time'
    filtered_dataframe.reset_index(inplace=True)
    filtered_dataframe['time'] = filtered_dataframe['time'].div(precision)
    filtered_dataframe.set_index('time', inplace=True)

    return filtered_dataframe


def main():
    parser = argparse.ArgumentParser(prog='resample.py',
                                     usage='resample.py file1 file2 .. -o output_file [-m method]')
    parser.add_argument(
        'input_files', type=argparse.FileType('r'), nargs='+', help='list of input files')
    parser.add_argument(
        '-o', dest='output_file', metavar='output_file', type=argparse.FileType('w'), 
        help='csv file with resampled output', required=True)
    parser.add_argument(
        '-m', dest='method', metavar='method', default='slinear', help='interpolation method')
    args = parser.parse_args()

    file_list = args.input_files
    output_file = args.output_file
    method = args.method
    frames = []

    for file in file_list:
        dataframe = pd.read_csv(file, header=None, names=COLUMNS, dtype=COLUMN_TYPES, skiprows=ROWS_TO_SKIP)
        resampled_dataframe = resample_dataframe(dataframe, max_value=3600, step=0.01, method=method)
        frames.append(resampled_dataframe)

    result = pd.concat(frames, axis=1)
    result.columns = [file.name for file in file_list]
    result.to_csv(output_file)


if __name__ == "__main__":
    main()
