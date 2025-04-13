import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def logistic_regression(df, algo):
    
    unique_houses = df['Hogwarts House'].drop_duplicates()
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    thetas = {}
    alpha = 0.1
    epsilon = 0.01

    # set x values
    x = df.select_dtypes(include=[np.number]).to_numpy()
    # set all the first values of x to the intercept value (1)
    x = np.insert(x, 0, np.full(df.shape[0], 1), axis=1)

    for house_name in unique_houses:
        # subtract one for house_name and add one for intercept
        thetas[house_name] = np.zeros(df.shape[1])
        # set a vector of y (1 if is the house where we are applying the one vs all)
        y = (df['Hogwarts House'] == house_name).astype(float).to_numpy()

        for t in range(1000):
            derivative = np.zeros(df.shape[1])
            h = sigmoid(np.dot(x, thetas[house_name]))

            # iterate rows
            for j in range(len(derivative)):
                sum = 0
                for i in range(df.shape[0]):
                    sum += (h[i] - y[i]) * x[i][j]
                derivative[j] = (1/df.shape[1]) * sum

            new_theta = thetas[house_name] - (alpha * derivative)
            if (np.linalg.norm(new_theta - thetas[house_name]) < epsilon):
                break
            thetas[house_name] = new_theta

        thetas[house_name] = new_theta.tolist()

    return (thetas)

def save_weights(df, thetas):
    f = open("weights", "w")
    f.write(json.dumps(thetas))
    f.write('\n')
    # save selected values
    f.write(json.dumps({'selected': list(df.loc[:, df.columns != 'Hogwarts House'])}))
    f.close()


def main(df):
    
    # drop NaN and Index
    df = df.drop('Index', axis=1)
    df = df.dropna()
    # get only Hogwarts House
    house_df = df['Hogwarts House']
    # get only numerical values
    df = df.select_dtypes(include=[np.number])

    try:

        #program_input = input("\nWHAT DATASET DO YOU WANT TO TRAIN THE MODEL WITH?:\n\t- ALL THE DATA (ALL)\n\t- SELECTED DATA (SEL)\n\n- ")    
        program_input = 'SEL'
        if program_input == "ALL":
            df
        elif program_input == "SEL":
            df = df[['Astronomy', 'Ancient Runes', 'Charms', 'Herbology', 'Defense Against the Dark Arts', 'Divination']]
        else:
            raise Exception("Please select an available dataset to train")

        #standarize df
        df = (df - df.mean()) / df.std()
        #join Hogwarts Houses names
        df = pd.concat([house_df, df], axis=1)

        #program_input = input("\nWHAT ALGORITHM DO YOU WANT TO USE?:\n\t- BATCH GRADIENT DESCENT (BGD)\n\t- STOCHASTIC GRADIENT DESCENT (SGD)\n\t- MINI-BATCH GRADIENT DESCENT (MBGD)\n\n- ")
        program_input = "BGD"
        if program_input == "BGD" or program_input == "SGD" or program_input == "MBGD":
            thetas = logistic_regression(df=df, algo=program_input)
        else:
            raise Exception("Please select an available algorithm")

        save_weights(df, thetas)

    except KeyboardInterrupt:
        return
    except Exception as e:
        sys.stderr.write(f'Error: {e}\n')
        sys.exit(1)
        return


if (__name__ == '__main__'):
    
    # error checking...
    try:
        df = pd.read_csv(sys.argv[1])
    except:
        sys.stderr.write('Error: Please enter a valid dataset')
        sys.exit(1)
    
    main(df)