from utils import anomalies_detection
from utils import prepare_dataset
import pandas as pd
import argparse

def main(args):

    filename=f'data/{args.data_type}_norm_confirmed/all_concatenated.csv'
    df=pd.read_csv(filename, sep='\t')
    print(df)
    #plots.plot_some_data(df)

    #searching for unnormal columns
    df_normality_scores, df_outliers = anomalies_detection.test_normality(filename, args.columns_to_drop)
    df_normality_scores.to_csv(f'results/{args.data_type}_outliers_values.csv', sep='\t', index=True)
    df_outliers.to_csv(f'results/{args.data_type}_outliers.csv', sep='\t', index=True)
    print(df_normality_scores)
    row_sum = df_normality_scores.loc['wynik testu'].sum()
    print("ilosc nienormalnych",row_sum)

    #delete unnormal columns
    folder=f'data/{args.data_type}_norm_confirmed'
    folder_out=f'data/{args.data_type}_norm_confirmed_normal'
    prepare_dataset.detele_unnormal_columns(folder, folder_out, df_normality_scores)

    #we sometimes want to test our model on the different dataset
    if args.test_data_type!="None":
        folder=f'data/{args.test_data_type}_norm_confirmed'
        folder_out=f'data/{args.test_data_type}_norm_confirmed_normal'
        prepare_dataset.detele_unnormal_columns(folder, folder_out, df_normality_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for deleting unnormal features")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--test_data_type", nargs="?", default="None", help="Type of test dataset based on norm_confirmed: positive/negative/all/None, choose None if you don't want to test on the different dataset", type=str)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume'], help="Columns to drop", type=list)
    args = parser.parse_args()
    main(args)