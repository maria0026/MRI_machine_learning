from utils import anomalies_detection
from utils import prepare_dataset
import pandas as pd
import argparse

def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    anomalies_detector = anomalies_detection.AnomaliesDetector()

    filename = f'data/{args.data_type}_{args.folder_in_end}/{args.filename}'
    df = pd.read_csv(filename, sep='\t')
    print(df)
    #plots.plot_some_data(df)

    #searching for unnormal columns
    df_normality_scores, df_outliers = anomalies_detector.test_normality(filename, args.columns_to_drop)
    df_normality_scores.to_csv(f'{args.results_directory}/{args.data_type}_outliers_values.csv', sep='\t', index=True)
    df_outliers.to_csv(f'{args.results_directory}/{args.data_type}_outliers.csv', sep='\t', index=True)
    print(df_normality_scores)
    row_sum = df_normality_scores.loc['wynik testu'].sum()
    print("ilosc nienormalnych",row_sum)

    #delete unnormal columns
    folder = f'data/{args.data_type}_{args.folder_in_end}'
    folder_out = f'data/{args.data_type}_{args.folder_out_end}'
    preprocessor.detele_unnormal_columns(folder, folder_out, df_normality_scores)

    #we sometimes want to test our model on the different dataset
    if args.test_data_type!="None":
        folder = f'data/{args.test_data_type}_{args.folder_in_end}'
        folder_out = f'data/{args.test_data_type}_{args.folder_out_end}'
        preprocessor.detele_unnormal_columns(folder, folder_out, df_normality_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for deleting unnormal features")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--test_data_type", nargs="?", default="None", help="Type of test dataset based on norm_confirmed: positive/negative/all/None, choose None if you don't want to test on the different dataset", type=str)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume'], help="Columns to drop", type=list)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--folder_in_end", nargs="?", default="norm_confirmed", help="End of folder in name", type=str)
    parser.add_argument("--folder_out_end", nargs="?", default="norm_confirmed_normal", help="End of folder out name", type=str)
    parser.add_argument("--filename", nargs="?", default="all_concatenated.csv", help="Name of the file for processing", type=str)
    args = parser.parse_args()
    main(args)