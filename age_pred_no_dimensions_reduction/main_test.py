import pandas as pd
import argparse
import joblib
from utils import test
import torch
import numpy as np
from utils import nn_data, nn_model

def main(args):

    df = pd.read_csv(f'data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/leave_out.csv', sep='\t')
    print(df.shape)
    identifier=df['identifier']
    df = df.drop(columns=args.columns_to_drop, errors='ignore')
    input_dim = df.shape[1]-1

    #model_path=f'models/{args.atlas}/{args.model_name}'
    model_path=f'models/{args.atlas}/{args.model_name}_{args.data_type}_valid_{args.valid}'
    results_directory=f'{args.results_directory}/{args.atlas}'
    mses, rmses, maes= [], [], []

    for i in range(0, args.nr_of_train):
        X_test=df.drop(columns=args.label_names)
        y_test=df[args.label_names]
        X_test_to_scale = X_test.drop(columns=args.column_to_copy)
        scaler = joblib.load(f'{model_path}/scaler_train_nr_{i}.pkl')
        X_test_scaled = scaler.transform(X_test_to_scale)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_to_scale.columns, index=X_test_to_scale.index)
        X_test = pd.concat([X_test_scaled_df, X_test[args.column_to_copy]], axis=1)
        y_test['identifier'] = identifier
        feature=args.label_names

        
        if args.model_name=='forest':
            rf= joblib.load( f'{model_path}/model_train_nr_{i}.pkl')
            mse, rmse, mae, results_df = test.random_forest_regression_model(X_test, y_test, feature, rf)



        elif args.model_name=="svm":
            clf= joblib.load( f'{model_path}/model_train_nr_{i}.pkl')
            if args.valid==1:
                z= joblib.load(f'models/{args.model_name}_z_train_nr_{i}.pkl')
            else:
                z=None
            mse, rmse, mae, results_df, feature_importance = test.svm_regression_model(X_test, y_test, clf, z=z, feature=feature, comp=False)
           

        elif args.model_name=='fnn':
            y_test[feature] = y_test[feature]/100
            model = nn_model.NeuralNetwork(input_dim, args.fnn_hidden_dim, args.output_dim)
            model.load_state_dict(torch.load(f'{model_path}/model_train_nr_{i}.pth', weights_only=True))
            mse, rmse, mae, results_df, feature_importance = test.neural_network_regression(X_test, y_test, args.batch_size, model,feature)

        elif args.model_name=='rnn':
            y_test[feature] = y_test[feature]/100
            model = nn_model.RNNModel(input_dim, args.rnn_hidden_dim, args.rnn_layer_dim, args.output_dim)
            model.load_state_dict(torch.load(f'{model_path}/model_train_nr_{i}.pth', weights_only=True))
            mse, rmse, mae, results_df = test.recurrent_neural_network_regression(X_test, y_test, args.batch_size, args.rnn_seq_dim, input_dim, model, feature)


        if i==0:
            results_df.to_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
            feature_importance.to_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}.csv', sep='\t')
        else:
            results_df_old = pd.read_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t')
            results_df = pd.concat([results_df_old, results_df], axis = 1)
            results_df.to_csv(f'{results_directory}/test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
            #change names of columns
            feature_importance.columns=[f'{col}_{i}' for col in feature_importance.columns]
            importance_df_old = pd.read_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}.csv', sep='\t', index_col=0)
            importance_df = pd.concat([importance_df_old, feature_importance], axis = 1)
            if i==args.nr_of_train-1:
                df_imp=importance_df.copy()
                feature_cols = [col for col in df_imp.columns if 'feature_name' not in col]
                df_imp = importance_df[feature_cols].copy()
                df_mean_std=df_imp.copy()
                df_mean_std['mean']=df_imp.mean(axis=1)
                df_mean_std['std']=df_imp.std(axis=1)
                print(importance_df)
                print("nowy", df_mean_std)
                df_mean_std['feature_name']=importance_df['feature_name']
                df_mean_std.sort_values(by='mean', ascending=False, inplace=True)
                importance_df=df_mean_std

            importance_df.to_csv(f'{results_directory}/test_{args.data_type}_importance_age_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=True)


        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)


    print("Mean squared error", np.mean(mses), np.std(mses))
    print("Root mean squared error", np.mean(rmses), np.std(rmses))
    print("Mean absolute error", np.mean(maes), np.std(maes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for age preidction")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--atlas", nargs="?", default="APARC", help="atlas", type=str)
    parser.add_argument("--valid", nargs="?", default=1, help="create valid set: 0/1", type=bool)
    parser.add_argument("--model_name", nargs="?", default="svm", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--test_one", nargs="?", default=0, help="Test one case", type=bool)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female', 'weight', 'hight'], help="Columns to drop", type=list)
    parser.add_argument("--division_by_total_volume", nargs="?", default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0", type=bool)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--column_to_copy", nargs="?", default=['male'], help="Columns to copy", type=list)
    parser.add_argument("--first_quantile", nargs="?", default=0.01, help="First quantile for svm regression", type=float)
    parser.add_argument("--last_quantile", nargs="?", default=0.99, help="Last quantile for svm regression", type=float)
    parser.add_argument("--batch_size", nargs="?", default=64, help="Batch size", type=int)
    parser.add_argument("--plot", nargs="?", default=1, help="Plot results", type=bool)
    parser.add_argument("--nr_of_train", nargs="?", default=5, help="Number of train dataset", type=int)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--fnn_momentum", nargs="?", default=0.7, help="Momentum for feed forward neural network", type=float)
    parser.add_argument("--fnn_weight_decay", nargs="?", default=0.01, help="Weight decay for feed forward neural network", type=float)
    parser.add_argument("--fnn_learning_rate", nargs="?", default=0.075, help="Learning rate for feed forward neural network", type=float)
    parser.add_argument("--rnn_learning_rate", nargs="?", default=1e-3, help="Learning rate for recurrent neural network", type=float)
    parser.add_argument("--fnn_hidden_dim", nargs="?", default=20, help="Hidden dimension for feed forward neural network", type=int)
    parser.add_argument("--rnn_hidden_dim", nargs="?", default=10, help="Hidden dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_layer_dim", nargs="?", default=1, help="Layer dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_seq_dim", nargs="?", default=1, help="Sequence dimension for recurrent neural network", type=int)
    parser.add_argument("--output_dim", nargs="?", default=1, help="Output dimension for neural network", type=int)
    args = parser.parse_args()
    main(args)
