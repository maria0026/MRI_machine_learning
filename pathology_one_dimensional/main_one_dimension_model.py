from utils import explain, prepare_dataset
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    df_features_all = pd.DataFrame()

    df_data = pd.read_csv(f'data/{args.test_data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    df_data = df_data.drop(columns=args.columns_to_drop)
    if args.division_by_total_volume:
        print("devided")
        df_data = preprocessor.divide_by_total_volume(df_data)

    df_identifiers = pd.read_csv(f'{args.results_directory}/train_{args.data_type}_test_{args.test_data_type}_identifiers_svm_valid_1.csv', sep='\t')
    df_female = pd.read_csv(f'{args.results_directory}/female_features_scores.csv', sep='\t', index_col=0)
    df_male = pd.read_csv(f'{args.results_directory}/male_features_scores.csv', sep='\t', index_col=0)

    #odfiltruj dane pathology outliers z all_concatenated na podstawie identfieiers
    for i in range (args.n_crosval):
        df_abnormalities_male_all = pd.DataFrame()
        df_abnormalities_female_all = pd.DataFrame()
      
        df_data_filtered = df_data[df_data['identifier'].isin(df_identifiers[f'identifier_lower_{i}']) | df_data['identifier'].isin(df_identifiers[f'identifier_upper_{i}'])]
        df_data_filtered = pd.concat([df_data_filtered, pd.DataFrame({'lower': df_data_filtered['identifier'].isin(df_identifiers[f'identifier_lower_{i}']).astype(int)})], axis=1)

        df_data_male = df_data_filtered[df_data_filtered['male'] == 1]
        df_data_female = df_data_filtered[df_data_filtered['male'] == 0]

        
        for column in df_data_filtered.columns:
            if column not in ['identifier', 'lower', 'male']:
                df_abnormalitites_male= explain.calculate_normality(df=df_data_male, column=column, X_feature_name=args.label_names, df_scores=df_male)
                df_abnormalities_female= explain.calculate_normality(df=df_data_female, column=column, X_feature_name=args.label_names, df_scores=df_female)

                df_abnormalities_male_all = pd.concat([df_abnormalities_male_all, df_abnormalitites_male], axis=1)
                df_abnormalities_female_all = pd.concat([df_abnormalities_female_all, df_abnormalities_female], axis=1)

        df_abnormalities_male_all.set_index(df_data_male['identifier'], inplace=True)
        df_data_male_new = df_data_male.set_index('identifier')
        df_abnormalities_male_all['lower']=df_data_male_new['lower']
        df_abnormalities_male_all['male'] = 1
        
    
        df_abnormalities_female_all.set_index(df_data_female['identifier'], inplace=True)
        df_data_female_new = df_data_female.set_index('identifier')
        df_abnormalities_female_all['lower']=df_data_female_new['lower']
        df_abnormalities_female_all['male'] = 0

        df_abnormalities_all = pd.concat([df_abnormalities_male_all, df_abnormalities_female_all], axis=0)
        
        df_features_results = explain.create_features_results(df_abnormalities_all)
        df_features_results['n_crosval'] = i
        print(df_features_results)
        df_features_all = pd.concat([df_features_all, df_features_results])
    

    df_features_all.to_csv(f'{args.results_directory}/{args.test_data_type}_features_results.csv', sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for features explainer")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--n_crosval", nargs="?", default=5, help="Number of crossvalidation", type=int)
    parser.add_argument("--test_data_type", nargs="?", default="negative_pathology", help="Type of test dataset based on norm_confirmed: positive/negative/all/None, choose None if you don't want to test on the different dataset", type=str)
    parser.add_argument("--division_by_total_volume", type=int, choices=[0, 1], default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0")
    parser.add_argument("--columns_to_drop", nargs="?", default=['norm_confirmed', 'sex', 'female'], help="Columns to drop", type=list)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    args = parser.parse_args()
    main(args)