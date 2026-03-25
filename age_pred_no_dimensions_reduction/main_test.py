import pandas as pd
import argparse
import joblib
from utils import test
import torch
import numpy as np
from utils import nn_model, prepare_dataset

def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    tester = test.ModelTester()
    mses, rmses, maes= [], [], []


    model_path=f'models/{args.atlas}/{args.model_name}_{args.data_type}_valid_{args.valid}'
    results_directory=f'{args.results_directory}/{args.atlas}'

    if 'big' in args.data_type:
        df = pd.read_csv(f'data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/leave_out_big.csv', sep=None, engine='python', dtype={'identifier': str})
    else:
        #df = pd.read_csv(f'data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/leave_out_{args.hearing_loss}.csv', sep=None, engine='python', dtype={'identifier': str})
        df = pd.read_csv(f'data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/leave_out.csv', sep=None, engine='python', dtype={'identifier': str})

    df = pd.read_csv(f'data/TW/mariawaligorska.csv', sep=',')

    print(df.columns)
    #cols_to_check = df.columns.difference(['L_HEARING_TYPE', 'P_HEARING_TYPE'])
    #df = df.dropna(subset=cols_to_check).copy()
    df = preprocessor.filter_age(df, args.label_names)
    df = preprocessor.filter_zeros(df, 10)

    '''
    df = df[df['IF_FIRST'] == 1]
    df['DATA_BADANIA_MRI'] = pd.to_datetime(df['DATA_BADANIA_MRI'], format='%Y-%m-%d')
    df['DATA_BADANIA'] = pd.to_datetime(df['DATA_BADANIA'], dayfirst=True, errors='coerce')
    df = df[df['DATA_BADANIA_MRI'] > df['DATA_BADANIA']]
    '''
    identifier=df['identifier']
    df = df.drop(columns=args.columns_to_drop, errors='ignore')
    input_dim = df.shape[1]-1
    #filter column which do not have A2009 in their name or age
    df = df[df.columns[df.columns.str.contains('A2009|age|male|hight|weight')]]

    for i in range(0, args.nr_of_train):
        X_test=df.drop(columns=args.label_names)
        y_test=df[args.label_names]
        numeric_cols = X_test.select_dtypes(include='number').columns
        cols_to_scale = numeric_cols.difference(args.column_to_copy)
        X_test_to_scale = X_test[cols_to_scale]
        print("Model path:", model_path)
        scaler = joblib.load(f'{model_path}/scaler_train_nr_{i}.pkl')
        X_test_scaled = scaler.transform(X_test_to_scale)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_to_scale.columns, index=X_test_to_scale.index)
        X_test = pd.concat([X_test_scaled_df, X_test[args.column_to_copy]], axis=1)
        y_test['identifier'] = identifier
        feature=args.label_names
        print(X_test.shape)

        features=['A2009-ctx-lh-G_occipital_middle_ThickStd', 'A2009-ctx-lh-S_circular_insula_inf_ThickAvg', 'A2009-ctx-lh-S_circular_insula_inf_ThickStd', 'A2009-ctx-rh-S_central_GrayVol', 'A2009-ctx-rh-S_central_ThickStd', 'A2009-ctx-rh-S_circular_insula_inf_GrayVol']
        #X_test=X_test[features]


        if args.model_name=='forest':
            rf= joblib.load( f'{model_path}/model_train_nr_{i}.pkl')
            mse, rmse, mae, results_df = tester.random_forest_regression_model(X_test, y_test, feature, rf)


        elif args.model_name=="svm":
            clf = joblib.load( f'{model_path}/model_train_nr_{i}.pkl')
            if args.valid==1:
                z = joblib.load(f'{model_path}/z_train_nr_{i}.pkl')
            else:
                z = None
            mse, rmse, mae, results_df, feature_importance, shap_values = tester.svm_regression_model(X_test, y_test, clf, z=z, feature=feature, comp=False, importance=True, shap_bool=args.shap)
            np.save(f'{model_path}/{args.model_name}_shap_values_{i}.npy', shap_values)

        elif args.model_name=='fnn':
            y_test[feature] = y_test[feature]/100
            model = nn_model.NeuralNetwork(input_dim, args.fnn_hidden_dim, args.output_dim)
            model.load_state_dict(torch.load(f'{model_path}/model_train_nr_{i}.pth', weights_only=True))
            mse, rmse, mae, results_df, feature_importance = tester.neural_network_regression(X_test, y_test, args.batch_size, model,feature)

        elif args.model_name=='rnn':
            y_test[feature] = y_test[feature]/100
            model = nn_model.RNNModel(input_dim, args.rnn_hidden_dim, args.rnn_layer_dim, args.output_dim)
            model.load_state_dict(torch.load(f'{model_path}/model_train_nr_{i}.pth', weights_only=True))
            mse, rmse, mae, results_df = tester.recurrent_neural_network_regression(X_test, y_test, args.batch_size, args.rnn_seq_dim, input_dim, model, feature)
        '''

        if i==0:
            results_df.to_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}_{args.hearing_loss}.csv', sep='\t', index=False)
            if args.model_name=='svm':
                feature_importance.to_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}_{args.hearing_loss}.csv', sep='\t')

        else:
            results_df_old = pd.read_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}_{args.hearing_loss}.csv', sep='\t')
            results_df = pd.concat([results_df_old, results_df], axis = 1)
            results_df.to_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}_{args.hearing_loss}.csv', sep='\t', index=False)
            if args.model_name=='svm':
                feature_importance.columns=[f'{col}_{i}' for col in feature_importance.columns]
                importance_df_old = pd.read_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}_{args.hearing_loss}.csv', sep='\t', index_col=0)
                importance_df = pd.concat([importance_df_old, feature_importance], axis = 1)

                if i==args.nr_of_train-1:
                    feature_cols = [col for col in importance_df.columns if 'feature_name' not in col]
                    df_mean_std = importance_df[feature_cols].copy()
                    df_mean_std['mean']=df_mean_std.mean(axis=1)
                    df_mean_std['std']=df_mean_std.std(axis=1)
                    df_mean_std['feature_name']=importance_df['feature_name']
                    df_mean_std.sort_values(by='mean', ascending=False, inplace=True)
                    importance_df=df_mean_std

                importance_df.to_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}_{args.hearing_loss}.csv', sep='\t', index=True)

            print("MAE:", mae)
        '''
        if i==0:
            results_df.to_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
            if args.model_name=='svm':
                feature_importance.to_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}.csv', sep='\t')

        else:
            results_df_old = pd.read_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t')
            results_df = pd.concat([results_df_old, results_df], axis = 1)
            results_df.to_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
            if args.model_name=='svm':
                feature_importance.columns=[f'{col}_{i}' for col in feature_importance.columns]
                importance_df_old = pd.read_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}.csv', sep='\t', index_col=0)
                importance_df = pd.concat([importance_df_old, feature_importance], axis = 1)

                if i==args.nr_of_train-1:
                    feature_cols = [col for col in importance_df.columns if 'feature_name' not in col]
                    df_mean_std = importance_df[feature_cols].copy()
                    df_mean_std['mean']=df_mean_std.mean(axis=1)
                    df_mean_std['std']=df_mean_std.std(axis=1)
                    df_mean_std['feature_name']=importance_df['feature_name']
                    df_mean_std.sort_values(by='mean', ascending=False, inplace=True)
                    importance_df=df_mean_std

                importance_df.to_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=True)

            print("MAE:", mae)



        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        print('mae:', mae)



    mae_mean = round(np.mean(maes), 2)
    mae_std = round(np.std(maes), 2)
    print("Mean squared error", np.mean(mses), np.std(mses))
    print("Root mean squared error", np.mean(rmses), np.std(rmses))
    print("Mean absolute error train", mae_mean, "± ", mae_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser for age prediction - testing on holdout set without dimensionality reduction")
    parser.add_argument("--model_name", nargs="?", default="svm", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--atlas", nargs="?", default="a2009", help="Atlas used for feature extraction", type=str)
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--hearing_loss", nargs="?", default="umiarkowany", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--valid", nargs="?", default=1, help="Create valid set and detrend: 0 (no) /1 (yes)", type=bool)
    parser.add_argument("--shap", nargs="?", default=1, help="calculate shap values", type=bool)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female', 'hight', 'weight'], help="Columns to drop", type=list)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    #parser.add_argument("--column_to_copy", nargs="?", default=['male', 'DATA_BADANIA', 'IF_FIRST'], help="Columns to copy", type=list)
    parser.add_argument("--column_to_copy", nargs="?", default=['male'], help="Columns to copy", type=list)
    parser.add_argument("--batch_size", nargs="?", default=64, help="Batch size", type=int)
    parser.add_argument("--nr_of_train", nargs="?", default=5, help="Number of train dataset", type=int)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--fnn_hidden_dim", nargs="?", default=20, help="Hidden dimension for feed forward neural network", type=int)
    parser.add_argument("--rnn_hidden_dim", nargs="?", default=10, help="Hidden dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_layer_dim", nargs="?", default=1, help="Layer dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_seq_dim", nargs="?", default=1, help="Sequence dimension for recurrent neural network", type=int)
    parser.add_argument("--output_dim", nargs="?", default=1, help="Output dimension for neural network", type=int)
    args = parser.parse_args()
    main(args)
