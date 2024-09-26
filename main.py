import pandas as pd
import numpy as np
from utils import dimensions_reduction, prepare_dataset, plots, train, valid, test, nn_data, test_one
import argparse
import joblib

def main(scaler):
    preprocessor = prepare_dataset.DatasetPreprocessor()

    df = pd.read_csv(f'data/{args.data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    identifier=df['identifier']
    df = df.drop(columns=args.columns_to_drop)
    if args.division_by_total_volume:
            df = preprocessor.divide_by_total_volume(df)

    X_test=df.drop(columns=args.label_names)
    y_test=df[args.label_names]

    X_test_to_scale = X_test.drop(columns=args.column_to_copy)

    scaler = joblib.load(f'models/{args.model_name}_scaler_train_nr_4.pkl')
    X_test_scaled = scaler.transform(X_test_to_scale)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_to_scale.columns, index=X_test_to_scale.index)
    X_test = pd.concat([X_test_scaled_df, X_test[args.column_to_copy]], axis=1)
    print(X_test)
    y_test['identifier'] = identifier

    pca_mri = joblib.load(f'models/{args.model_name}_pca_mri_train_nr_4.pkl')
    test_pca = pca_mri.transform(X_test)

    X_test = pd.DataFrame(data = test_pca
            , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)
    
    feature=args.label_names
    clf= joblib.load( f'models/{args.model_name}_model_train_nr_4.pkl')
    z= joblib.load(f'models/{args.model_name}_z_train_nr_4.pkl')
    z_quantiles=joblib.load(f'models/{args.model_name}_z_quantiles_train_nr_4.pkl')
    identifiers_list, lower_outliers, upper_outliers=test_one.svm_regression(clf, X_test, y_test, z=z, z_quantiles=z_quantiles, feature=feature, plot=args.plot, first_quantile=args.first_quantile, last_quantile=args.last_quantile)
    print(identifiers_list, lower_outliers, upper_outliers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for age preidction")
    parser.add_argument("--data_type", nargs="?", default="negative", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--model_name", nargs="?", default="svm", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--test_one", nargs="?", default=0, help="Test one case", type=bool)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female'], help="Columns to drop", type=list)
    parser.add_argument("--division_by_total_volume", nargs="?", default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0", type=bool)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--column_to_copy", nargs="?", default=['male'], help="Columns to copy", type=list)
    parser.add_argument("--first_quantile", nargs="?", default=0.01, help="First quantile for svm regression", type=float)
    parser.add_argument("--last_quantile", nargs="?", default=0.99, help="Last quantile for svm regression", type=float)
    parser.add_argument("--plot", nargs="?", default=1, help="Plot results", type=bool)
    args = parser.parse_args()
    #scaler
    #pca_mri
    #clf
    #x_quantiles
    main(args)
