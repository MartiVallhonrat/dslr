import sys
import pandas as pd
import numpy as np
import math

def count(column):

    count_result = 0
    for i in range(len(column)):
        count_result += 1

    return (count_result)


def min(column):

    for i in range(len(column)):

        if (i == 0):
            min_result = column.iloc[i]
        else:
            min_result = column.iloc[i] if column.iloc[i] < min_result else min_result
    
    return (min_result)

def max(column):

    for i in range(len(column)):

        if (i == 0):
            max_result = column.iloc[i]
        else:
            max_result = column.iloc[i] if column.iloc[i] > max_result else max_result
    
    return (max_result)

def percentile(column, percent):

    column = column.sort_values()
    # 0 index so -1
    pos = ((percent / 100) * (len(column) - 1.0))
    diff = pos - math.floor(pos)
    
    if (diff != 0):
        result_per = column.iloc[math.floor(pos)] + ((column.iloc[math.ceil(pos)] - column.iloc[math.floor(pos)]) * diff)
    else:
        result_per = column.iloc[int(pos)]

    return (result_per)


def main(input_df):

    # just to check
    print(input_df.describe())

    # drop Index column
    input_df = input_df.drop('Index', axis=1)
    # get only numerical values
    input_df = input_df.select_dtypes(include=[np.number])
    # drop nan columns
    input_df = input_df.dropna(axis=1, how='all')

    # define lambdas
    mean = lambda column: column.sum() / len(column)
    std = lambda column: math.sqrt(sum([(x - mean(column))**2 for x in column]) / (len(column) - 1))

    output_df = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])

    for column_name in input_df:
        column = input_df[column_name].dropna()
        output_df[column_name] = [
            count(column),
            mean(column),
            std(column),
            min(column),
            percentile(column, 25),
            percentile(column, 50),
            percentile(column, 75),
            max(column),
        ]
    
    print(output_df)

if (__name__ == '__main__'):

    # error checking...
    if (len(sys.argv) < 2):
        sys.stderr.write('Error: Please enter the dataset as argument\n')
        sys.exit(1)

    try:
        input_df = pd.read_csv(sys.argv[1])
    except:
        sys.stderr.write('Error: Invalid dataset path\n')
        sys.exit(1)

    main(input_df)