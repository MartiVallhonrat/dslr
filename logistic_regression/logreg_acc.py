import sys
import pandas as pd
import numpy as np
import json
from logreg_predict import classify_data

def calculate_acc(val_df, predict_df):
    correct_predict = 0
    incorrect_predict = 0
    print('----------ACCURACY TEST----------\n')
    for i in range(len(predict_df)):
        if (predict_df.iloc[i]['Hogwarts House'] == val_df.iloc[i]['Hogwarts House']):
            correct_predict += 1
        else:
            if (incorrect_predict == 0):
                print("INCORRECT PREDICTIONS:")
            print(f"\t- {predict_df.iloc[i]['Index']}:  {predict_df.iloc[i]['Hogwarts House']} != {val_df.iloc[i]['Hogwarts House']}")
            incorrect_predict += 1
    print(f"\nTOTAL INCORRECT: {correct_predict}")
    print(f"TOTAL CORRECT: {incorrect_predict}")
    print(f"ACCURACY: {correct_predict / (correct_predict + incorrect_predict) * 100}")


def main(thetas, df):
    num_df = df.select_dtypes(include=[np.number])
    # standarize df
    num_df = (num_df - num_df.mean()) / num_df.std()
    predict_df = classify_data(thetas, num_df)
    calculate_acc(val_df=df, predict_df=predict_df)


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
        selected.insert(0, 'Hogwarts House')
        df = df[selected]
        df = df.dropna()
    except Exception as e:
        sys.stderr.write(f'Error: {e}\n')
        sys.exit(1)
    
    main(thetas, df)