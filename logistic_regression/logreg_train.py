import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def logistic_regression(df, algo):
    
    unique_houses = df['Hogwarts House'].drop_duplicates()

    for house_name in unique_houses:
        print(f"DOING LOGISTIC REGRESSION OF {house_name} WITH {algo}...")


def main(df):
    
    # drop NaN and Index
    df = df.drop('Index', axis=1)
    df = df.dropna()
    # get only Hogwarts House
    house_df = df['Hogwarts House']
    # get only numerical values
    df = df.select_dtypes(include=[np.number])

    try:

        program_input = input("\nWHAT DATASET DO YOU WANT TO TRAIN THE MODEL WITH?:\n\t- ALL THE DATA (ALL)\n\t- SELECTED DATA (SEL)\n\n- ")    
        if program_input == "ALL":
            df
        elif program_input == "SEL":
            df = df[['Astronomy', 'Ancient Runes', 'Charms', 'Herbology', 'Defense Against the Dark Arts', 'Divination']]
        else:
            raise Exception("Please select an available dataset to train")

        # standarize df
        df = (df - df.mean()) / df.std()
        # join Hogwarts Houses names
        df = pd.concat([house_df, df], axis=1)

        program_input = input("\nWHAT ALGORITHM DO YOU WANT TO USE?:\n\t- BATCH GRADIENT DESCENT (BGD)\n\t- STOCHASTIC GRADIENT DESCENT (SGD)\n\t- MINI-BATCH GRADIENT DESCENT (MBGD)\n\n- ")
        if program_input == "BGD" or program_input == "SGD" or program_input == "MBGD":
            thetas = logistic_regression(df=df, algo=program_input)
        else:
            raise Exception("Please select an available algorithm")

    except KeyboardInterrupt:
        return
    except Exception as e:
        sys.stderr.write(f'Error: {e}\n')
        sys.exit(1)
        return


if (__name__ == '__main__'):
    
    # error checking...
    try:
        df = pd.read_csv('../datasets/dataset_train.csv')
    except:
        sys.stderr.write('Error: Please locate the dataset like the path: "../datasets/dataset_train.csv"')
        sys.exit(1)
    
    main(df)