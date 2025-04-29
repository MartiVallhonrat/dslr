import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def main(thetas, df):

    # get only Hogwarts Hous
    house_df = df['Hogwarts House']
    unique_house = house_df.drop_duplicates().to_numpy()
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow',
    }

    # get only numerical values
    df = df.select_dtypes(include=[np.number])
    # standarize df
    df = (df - df.mean()) / df.std()
    sigmod = lambda z: 1 / (1 + np.exp(-z))
    
    num_houses = len(unique_house)
    num_features = len(df.columns)
    # generate subplots
    fig, axes = plt.subplots(num_houses, num_features, figsize=(30, 20))

    for i in range(num_houses):
        house_name = unique_house[i]
        for j in range(num_features):

            ax = axes[i, j]
            
            # draw values
            join_df = pd.concat([house_df, df.iloc[:, j]], axis=1)
            x1 = join_df.loc[join_df['Hogwarts House'] == house_name].iloc[:, 1]
            x2 = join_df.loc[join_df['Hogwarts House'] != house_name].iloc[:, 1]
            y1 = np.full(len(x1), 1)
            y2 = np.full(len(x2), 0)

            ax.scatter(x1, y1, color=colors[house_name], alpha=0.5)
            ax.scatter(x2, y2, color='grey', alpha=0.5)

            # draw sigmod
            x = np.linspace(-4, 4, 100)
            # adding one two skip intercept
            w = thetas[house_name][j + 1]
            y = sigmod(w * x)
            ax.plot(x, y)

            if i == (num_houses - 1):
                ax.set_xlabel(df.columns[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(house_name, fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    try:
        plt.savefig('sigmoids_plot')
    except:
        sys.stderr.write('Error: Unable to store pair plot, path has to be: "sigmoids_plot"\n')
        sys.exit(1)

if (__name__ == '__main__'):
    
    if (len(sys.argv) < 3):
        sys.stderr.write('Error: Please enter the weights file and the csv in which you trained the model\n')
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
    except:
        sys.stderr.write('Error: Please enter a valid weights file and a valid predict csv\n')
        sys.exit(1)
    
    main(thetas, df)