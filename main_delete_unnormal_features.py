import pandas as pd
import argparse
from utils import anomalies_detection, prepare_dataset


def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    anomalies_detector = anomalies_detection.AnomaliesDetector()

    filename = f'data/{args.data_type}_norm_confirmed/{args.filename}'

    #searching for unnormal columns
    df_normality_scores, df_outliers = anomalies_detector.test_normality(filename, args.columns_to_drop)
    df_normality_scores.to_csv(f'{args.results_directory}/{args.data_type}_outliers_values.csv', sep='\t', index=True)
    df_outliers.to_csv(f'{args.results_directory}/{args.data_type}_outliers.csv', sep='\t', index=True)
    #print(df_normality_scores)
    row_sum = df_normality_scores.loc['wynik testu'].sum()
    print("ilosc nienormalnych",row_sum)

    #delete unnormal columns
    folder = f'data/{args.data_type}_norm_confirmed'
    folder_out = f'data/{args.data_type}_norm_confirmed_normal'
    preprocessor.detele_unnormal_columns(folder, folder_out, df_normality_scores)

    #we sometimes want to test our model on the different dataset
    if args.test_data_type!="None":
        folder = f'data/{args.test_data_type}_norm_confirmed'
        folder_out = f'data/{args.test_data_type}_norm_confirmed_normal'
        preprocessor.detele_unnormal_columns(folder, folder_out, df_normality_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for deleting unnormal features")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--test_data_type", nargs="?", default="None", help="Type of test dataset based on norm_confirmed: positive/negative/all/None, choose None if you don't want to test on the different dataset", type=str)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume'], help="Columns to drop", type=list)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--filename", nargs="?", default="all_concatenated.csv", help="Name of the file for processing", type=str)
    args = parser.parse_args()
    main(args)