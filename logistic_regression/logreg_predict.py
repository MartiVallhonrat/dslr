import sys
import pandas as pd
import numpy as np
import json

def classify_data(thetas, df):
    # add the intercept
    df.insert(0, 'X', np.full(df.shape[0], 1))
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    indexes = [0] * df.shape[0]
    predictions = [''] * df.shape[0]

    for i in range(df.shape[0]):
        row = df.iloc[i].to_numpy()
        indexes[i] = df.index[i]
        max_like = 0
        for house_name in thetas:
            house_like = sigmoid(np.dot(thetas[house_name], row))
            if house_like > max_like:
                predictions[i] = house_name
                max_like = house_like

    predict_data = {
        "Index": indexes,
        "Hogwarts House": predictions
    }
    predict_df = pd.DataFrame(predict_data)
    return (predict_df)


def main(thetas, df):
    predict_df = classify_data(thetas, df)
    try:
        predict_df.to_csv('houses.csv', index=False)
    except:
        sys.stderr.write("Error: Can't open 'house.csv' and write on it\n")
        sys.exit(1)


if (__name__ == '__main__'):
    
    if (len(sys.argv) < 3):
        sys.stderr.write('Error: Please enter the weights file and the dataset to predict as argument\n')
        sys.exit(1)

    try:
        # get weights and prepare them
        f = open(sys.argv[1], 'r')
        thetas = f.readline()
        thetas = json.loads(thetas)
        # transform to numpy arrays
        for house in thetas:
            thetas[house] = np.array(thetas[house])
        # get data to predict and prepare it
        df = pd.read_csv(sys.argv[2])
        # get selected
        selected = f.readline()
        selected = json.loads(selected)['selected']
        df = df[selected]
        df = df.dropna()
        # standarize df
        df = (df - df.mean()) / df.std()
    except:
        sys.stderr.write('Error: Please enter a valid weights file and a valid predict csv\n')
        sys.exit(1)
    
    main(thetas, df)