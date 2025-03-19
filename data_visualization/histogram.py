import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main(df):

    # get only numerical Hogwarts House
    house_df = df['Hogwarts House']
    unique_house = house_df.drop_duplicates()
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow',
    }
    # get only numerical values
    num_df = df.drop('Index', axis=1)
    num_df = num_df.select_dtypes(include=[np.number])
    
    for column_name in num_df:

        join_df = pd.concat([house_df, num_df[column_name]], axis=1)
        join_df = join_df.dropna()
        
        for house_name in unique_house:
            hist_list = join_df.loc[join_df['Hogwarts House'] == house_name]
            hist_list = hist_list[column_name]

            house_color = colors[house_name]
            plt.hist(hist_list, color=house_color, alpha=0.5)

        plt.title(column_name)
        plt.legend(unique_house)
        plt.xlabel('Grades')
        plt.ylabel('Quantity of students')
        try:
            plt.savefig(f'./graphs/histograms/{column_name}')
        except:
            sys.stderr.write(f'Error: Unable to store {column_name}, path has to be: "./graphs/histograms/"\n')
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