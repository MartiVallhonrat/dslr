import sys
import pandas as pd
import numpy as np
import math

def std(column):
    mean = lambda column: column.sum() / len(column)

    column = column.dropna()
    std_result = math.sqrt(sum([(x - mean(column))**2 for x in column]) / len(column))

    return (std_result)

def main(input_df):

    # drop Index column
    input_df = input_df.drop('Index', axis=1)
    # get only numerical values
    input_df = input_df.select_dtypes(include=[np.number])
    # drop nan columns
    input_df = input_df.dropna(axis=1, how='all')

    # define lambdas
    mean = lambda column: column.sum() / len(column)
    quarter = lambda column: (column.max() - column.min()) * 0.25
    half = lambda column: (column.max() - column.min()) * 0.5
    three_quarters = lambda column: (column.max() - column.min()) * 0.75

    output_df = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])

    for column in input_df:
        output_df[column] = [
            input_df[column].count(),
            mean(input_df[column]),
            std(input_df[column]),
            input_df[column].min(),
            quarter(input_df[column]),
            half(input_df[column]),
            three_quarters(input_df[column]),
            input_df[column].max(),
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