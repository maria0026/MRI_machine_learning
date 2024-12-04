from utils import explain, prepare_dataset
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    df = pd.read_csv(f'data/{args.data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    args.calculate_new_scores = bool(args.calculate_new_scores)
    #map sex
    sex_bool=0
    if args.sex=='male':
        sex_bool=1
    elif args.sex=='female':
        sex_bool=0

    df = df[df['male'] == sex_bool]
    df = df.drop(columns=args.columns_to_drop)

    df_scores = pd.DataFrame()
    df_abnormalities_all = pd.DataFrame()

    if args.division_by_total_volume:
        print("devided")
        df = preprocessor.divide_by_total_volume(df)
    if args.data_type == 'positive':
        df_train, df_test = train_test_split(df, test_size=0.3)
    else:
        df_test = df

    if args.calculate_new_scores is True:
        for column in df_train.columns:
            if column != 'identifier':
                model, coefficients, intercept = explain.calculate_trends(df_train, column, args.label_names)
                white_test_pvalue=explain.white_test(df_train, column, args.label_names, model)
                mean, std, skewness, kurt, mad, median, iqr, min_value, max_value = explain.scores(df_train, column, args.label_names, model)

                table_of_columns = ['intercept', '1', 'x', 'x^2', 'White test p-value', 'mean', 'std', 'skewness', 'kurtosis', 'mad', 'median', 'iqr', 'min', 'max']
                table_of_scores = [[intercept, coefficients[0], coefficients[1], coefficients[2], white_test_pvalue, mean, std, skewness, kurt, mad, median, iqr, min_value, max_value]]

                quantiles_array, indices_array = explain.calculate_quantiles(df_train, column, args.label_names, args.quantiles, model)
                table_of_columns.extend(indices_array)
                table_of_scores[0].extend(quantiles_array)
                
                df_column_scores = pd.DataFrame(data=table_of_scores, columns=table_of_columns)
                df_scores = pd.concat([df_scores, df_column_scores], axis=0)

        df_scores.set_index((df.drop(columns=['identifier'])).columns, inplace=True)
        df_scores.to_csv(f'{args.results_directory}/{args.sex}_features_scores.csv', sep='\t', index=True)


    #test
    df_scores = pd.read_csv(f'{args.results_directory}/{args.sex}_features_scores.csv', sep='\t', index_col=0)

    for column in df_test.columns:
        if column != 'identifier':
            df_abnormalitites= explain.calculate_normality(df_test, column, args.label_names, df_scores)
            df_abnormalities_all = pd.concat([df_abnormalities_all, df_abnormalitites], axis=1)

    df_abnormalities_all.set_index(df_test['identifier'], inplace=True)
    df_abnormalities_all.to_csv(f'{args.results_directory}/{args.sex}_{args.data_type}_features_abnormalities.csv', sep='\t', index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for features explainer")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--calculate_new_scores", type=int, choices=[0, 1], default=1, help="Calculate new scores (1 = True, 0 = False)")
    parser.add_argument("--sex", nargs="?", default='male', help="Select sex (male/female)", type=str)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--columns_to_drop", nargs="?", default=['norm_confirmed', 'sex', 'female', 'male'], help="Columns to drop", type=list)
    parser.add_argument("--division_by_total_volume", type=int, choices=[0, 1], default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0")
    parser.add_argument("--first_quantile", nargs="?", default=0.01, help="First quantile for svm regression", type=float)
    parser.add_argument("--last_quantile", nargs="?", default=0.99, help="Last quantile for svm regression", type=float)
    parser.add_argument("--plot", type=int, choices=[0, 1], default=1, help="Plot results (1 = True, 0 = False)")
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--quantiles", nargs="?", default=[0.005, 0.01, 0.02, 0.05], help="Quantiles for scores", type=list)
    args = parser.parse_args()
    main(args)
