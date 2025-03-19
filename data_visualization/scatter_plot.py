import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main(df):

    # get only numerical values
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
    sub_df = df
    
    for column_name in df:
        sub_df = sub_df.drop(columns=[column_name])
        for subcolumn_name in sub_df:
            plt.title(f'{column_name} X {subcolumn_name}')
            plt.xlabel(column_name)
            plt.ylabel(subcolumn_name)

            for house_name in unique_house:
                join_df = pd.concat([house_df, df[column_name], sub_df[subcolumn_name]], axis=1)
                list = join_df.loc[join_df['Hogwarts House'] == house_name]
                plt.scatter(list[column_name], list[subcolumn_name], color=colors[house_name], alpha=0.5)

            plt.legend(unique_house)
            
            try:
                plt.savefig(f'./graphs/scatter_plots/{column_name}_X_{subcolumn_name}')
            except:
                sys.stderr.write(f'Error: Unable to store {column_name}_X_{subcolumn_name}, path has to be: "./graphs/scatter_plots/"\n')
                sys.exit(1)
            
            plt.close('all')

if (__name__ == '__main__'):
    
    # error checking...
    try:
        df = pd.read_csv('../datasets/dataset_train.csv')
    except:
        sys.stderr.write('Error: Please locate the dataset like the path: "../datasets/dataset_train.csv"')
        sys.exit(1)
    
    main(df)