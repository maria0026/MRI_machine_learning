from utils import explain, prepare_dataset
import pandas as pd
import argparse


def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()

    df = pd.read_csv(f'data/{args.data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    df = df.drop(columns=args.columns_to_drop)

    table_of_scores = []
    table_of_indices = []
    quantiles = [0.005, 0.01, 0.02, 0.05]
    df_scores = pd.DataFrame()

    if args.division_by_total_volume:
        df = preprocessor.divide_by_total_volume(df)

    for column in df.columns:
        model, coefficients, intercept = explain.calculate_trends(df, column, args.label_names)
        white_test_pvalue=explain.white_test(df, column, args.label_names, model)
        mean, std, skewness, kurt, mad, median, iqr, min_value, max_value = explain.scores(df, column, args.label_names, model)

        table_of_scores = [intercept, coefficients[1], coefficients[2], coefficients[3], white_test_pvalue, mean, std, skewness, kurt, mad, median, iqr, min_value, max_value]
        table_of_indices = ['intercept', 'x', 'x^2', 'x^3', 'White test p-value', 'mean', 'std', 'skewness', 'kurtosis', 'mad', 'median', 'iqr', 'min', 'max']
        
        quantiles_array, indices_array = explain.calculate_quantiles(df, column, args.label_names, quantiles, model)
        table_of_scores.extend(quantiles_array)
        table_of_indices.extend(indices_array)

        df_column_scores = pd.DataFrame({column: table_of_scores}, index=table_of_indices)
        df_scores = pd.concat([df_scores, df_column_scores], axis=1)

    df_scores.to_csv(f'{args.results_directory}/features_scores.csv', sep='\t', index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for features explainer")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female', 'male'], help="Columns to drop", type=list)
    parser.add_argument("--division_by_total_volume", nargs="?", default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0", type=bool)
    parser.add_argument("--first_quantile", nargs="?", default=0.01, help="First quantile for svm regression", type=float)
    parser.add_argument("--last_quantile", nargs="?", default=0.99, help="Last quantile for svm regression", type=float)
    parser.add_argument("--plot", nargs="?", default=1, help="Plot results", type=bool)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    args = parser.parse_args()
    main(args)
