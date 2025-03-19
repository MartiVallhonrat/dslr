import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main(df):

    # drop NaN and Index
    df = df.drop('Index', axis=1)
    df = df.dropna()

    # get only Hogwarts House
    house_df = df['Hogwarts House']
    unique_house = house_df.drop_duplicates()
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow',
    }

    # get only numerical values
    df = df.select_dtypes(include=[np.number])

    # generate subplots
    num_features = len(df.columns)
    fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20))

    for i in range(num_features):
        for j in range(num_features):

            ax = axes[i, j]
            
            for house_name in unique_house:
                join_df = pd.concat([house_df, df.iloc[:, i], df.iloc[:, j]], axis=1)
                list = join_df.loc[join_df['Hogwarts House'] == house_name]

                if i == j:
                    ax.hist(list.iloc[:, 1], color=colors[house_name], alpha=0.5)
                else:
                    ax.scatter(list.iloc[:, 2], list.iloc[:, 1], color=colors[house_name], alpha=0.5)

            if j == 0:
                ax.set_ylabel(df.columns[i], fontsize=10)
            if i == (num_features - 1):
                ax.set_xlabel(df.columns[j], fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    try:
        plt.savefig(f'./graphs/pair_plot/pair_plot')
    except:
        sys.stderr.write(f'Error: Unable to store pair_plot, path has to be: "./graphs/pair_plot/"\n')
        sys.exit(1)

if (__name__ == '__main__'):
    
    # error checking...
    try:
        df = pd.read_csv('../datasets/dataset_train.csv')
    except:
        sys.stderr.write('Error: Please locate the dataset like the path: "../datasets/dataset_train.csv"')
        sys.exit(1)
    
    main(df)