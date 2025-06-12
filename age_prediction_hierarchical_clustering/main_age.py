import pandas as pd
import numpy as np
from torch import nn
import argparse
from scipy.stats import randint, uniform
import json
import joblib
import os
import torch
from sklearn.model_selection import train_test_split
from utils import prepare_dataset, train, valid, test, nn_data, dimensions_reduction



def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    trainer = train.ModelTrainer()
    tester = test.ModelTester()
    reductor = dimensions_reduction.DimensionsReductor()

    mses, rmses, maes = [], [], []
    loss_fn = nn.MSELoss()

    #path for saving model parameters
    model_path=f'models/{args.atlas}/{args.model_name}_{args.data_type}_valid_{args.valid}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    df = pd.read_csv(f'data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/all_concatenated.csv', sep='\t')
    #create leave out dataset
    if os.path.exists("data/leave_out_identifiers.csv"):
        leave_ids = pd.read_csv("data/leave_out_identifiers.csv")['identifier']
        df_leave = df[df['identifier'].isin(leave_ids)]
        df = df[~df['identifier'].isin(leave_ids)]
    else:
        df, df_leave = train_test_split(df, test_size=0.15, random_state=42)
        df_leave['identifier'].to_csv("data/leave_out_identifiers.csv", index=False)
    df_leave.to_csv(f'data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/leave_out.csv', sep='\t', index=False)

    identifier=df['identifier']
    df = df.drop(columns=args.columns_to_drop)
    input_dim = df.shape[1]-1

    #different dataset for testing
    if args.test_data_type!="None":
        df_test = pd.read_csv(f'data/{args.test_data_type}_norm_confirmed/all_concatenated.csv', sep='\t')
        identifier=df_test['identifier']
        df_test = df_test.drop(columns=args.columns_to_drop)


    for i in range(args.n_crosval):
        '''
        if args.division_by_total_volume:
            df = preprocessor.divide_by_total_volume(df)
            if args.test_data_type!="None":
                df_test = preprocessor.divide_by_total_volume(df_test)
        '''
        feature=args.label_names
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(df, args.label_names, test_size=args.test_size, valid=args.valid)


        if args.test_data_type!="None":
            X_test = df_test.drop(columns=args.label_names)
            y_test = df_test[args.label_names]

        X_train, X_val, X_test, scaler = preprocessor.standardize_data(X_train, X_val, X_test, column_to_copy=args.column_to_copy)
        joblib.dump(scaler, f'{model_path}/scaler_train_nr_{i}.pkl')
        y_test['identifier'] = identifier
        y_test['male']=X_test['male']
        print("Odchylenie",np.std(y_train[feature[0]]))
        print("Srednia", np.mean(y_train[feature[0]]))
        print("X_train", X_train.shape)
        print("X_test", X_test.shape)

        if args.model_name=='forest':
            forest_param_dist = json.loads(args.forest_param_dist)
            forest_param_dist['n_estimators'] = randint(*forest_param_dist['n_estimators'])
            forest_param_dist['max_depth'] = randint(*forest_param_dist['max_depth'])
            forest_param_dist['min_samples_leaf'] = randint(*forest_param_dist['min_samples_leaf'])
            rf = trainer.random_forrest_regression_model(X_train, y_train, forest_param_dist, *feature)
            mse, rmse, mae, results_df = tester.random_forest_regression_model(X_test, y_test, feature, rf)
            joblib.dump(rf, f'{model_path}/model_train_nr_{i}.pkl')


        elif args.model_name=="svm":
            svm_param_dist = json.loads(args.svm_param_dist)
            svm_param_dist['C'] = randint(*svm_param_dist['C'])  
            svm_param_dist['gamma'] = uniform(*svm_param_dist['gamma']) 
            

            y_train_h=y_train.copy()
            y_train_h['identifier'] = identifier
            y_train_h['male']=X_train['male']
            rng = np.random.default_rng(42)  # for reproducibility
            #selected_columns = rng.choice(X_train.columns, size=150, replace=False)

            # Reduce X_train to n features
            #X_train_h = X_train[selected_columns]
            X_train_h = X_train
        
            features=reductor.hierarchical_feature_selection(X_train_h, y_train_h, trainer, tester, args.model_name, svm_param_dist, feature)
            print("Selected features", features)
            X_train_selected = X_train[features]
            X_val_selected = X_val[features]
            X_test_selected = X_test[features]

            clf = trainer.svm_regression_model(X_train_selected, y_train, svm_param_dist, feature)

            if args.valid:
                z, z_quantiles= valid.svm_regression_model(X_val_selected, y_val, clf, feature, plot=args.plot)
            else:
                z=None
                z_quantiles=None
    
            mse, rmse, mae, results_df, feature_importance = tester.svm_regression_model(X_test_selected, y_test, clf, z=z, feature=feature)
            joblib.dump(clf, f'{model_path}/hierachical_model_train_nr_{i}.pkl')
            joblib.dump(z, f'{model_path}/hierachical_z_train_nr_{i}.pkl')
            joblib.dump(z_quantiles, f'{model_path}/hierachical_z_quantiles_train_nr_{i}.pkl')


        elif args.model_name=='fnn':
            y_train[feature] = y_train[feature]/100
            y_test[feature] = y_test[feature]/100
            train_dataloader = nn_data.load_fnn_data(X_train, y_train, args.batch_size, feature)
            model = trainer.feed_forward_neural_network(train_dataloader, input_dim, args.fnn_hidden_dim, args.output_dim, args.fnn_learning_rate, loss_fn, args.num_epochs, args.fnn_momentum, args.fnn_weight_decay)
            mse, rmse, mae, results_df, feature_importance = tester.neural_network_regression(X_test, y_test, args.batch_size, model,feature)
            torch.save(model.state_dict(), f'{model_path}/model_train_nr_{i}.pth')
            

        elif args.model_name=='rnn':
            y_train[feature] = y_train[feature]/100
            y_test[feature] = y_test[feature]/100
            train_dataloader = nn_data.load_rnn_data(X_train, y_train, args.batch_size, feature)
            model = trainer.recurrent_neural_network(train_dataloader, args.rnn_seq_dim, input_dim, args.rnn_hidden_dim, args.rnn_layer_dim, args.output_dim, args.rnn_learning_rate, loss_fn, args.num_epochs, args.rnn_weight_decay)
            mse, rmse, mae, results_df = tester.recurrent_neural_network_regression(X_test, y_test, args.batch_size, args.rnn_seq_dim, input_dim, model, feature)
            torch.save(model.state_dict(), f'{model_path}/model_train_nr_{i}.pth')

        results_directory=f'{args.results_directory}/{args.atlas}'
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        if i==0:
            results_df.to_csv(f'{results_directory}/train_{args.data_type}_test_{args.test_data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
        else:
            results_df_old = pd.read_csv(f'{results_directory}/train_{args.data_type}_test_{args.test_data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t')
            results_df = pd.concat([results_df_old.reset_index(drop = True), results_df.reset_index(drop = True)], axis = 1)
            results_df.to_csv(f'{results_directory}/train_{args.data_type}_test_{args.test_data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)


        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        print("MAE:", mae)

    mae_mean = round(np.mean(maes), 2)
    mae_std = round(np.std(maes), 2)
    print("Mean squared error", np.mean(mses), np.std(mses))
    print("Root mean squared error", np.mean(rmses), np.std(rmses))
    print("Mean absolute error train", mae_mean, "± ", mae_std)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser for age preidction - testing on test set without dimensionality reduction")
    parser.add_argument("--atlas", nargs="?", default="ASEG", help="atlas", type=str)
    parser.add_argument("--model_name", nargs="?", default="svm", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--valid", nargs="?", default=1, help="create valid set: 0/1", type=bool)
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--test_size", nargs="?", default=0.2, help="Size of test dataset", type=float)
    parser.add_argument("--test_data_type", nargs="?", default="None", help="Type of test dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--sex_subset", nargs="?", default="all", help="Choose the sex subset: all/female/male", type=str)
    parser.add_argument("--division_by_total_volume", nargs="?", default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0", type=bool)
    parser.add_argument("--n_most_important_features", nargs="?", default=20, help="Choose the number of extracting features that load into components")
    parser.add_argument("--n_crosval", nargs="?", default=5, help="Number of crossvalidation", type=int)
    parser.add_argument("--batch_size", nargs="?", default=64, help="Batch size", type=int)
    parser.add_argument("--num_epochs", nargs="?", default=100, help="Number of epochs", type=int)
    parser.add_argument("--forest_param_dist", nargs="?", default=json.dumps({
        'n_estimators': [50, 500],
        'max_depth': [1, 30],
        'min_samples_leaf': [5, 10],
        'min_samples_split': [10, 20, 30, 50, 80]}),
        help="JSON string for random forest parameter distribution")
    parser.add_argument("--svm_param_dist", nargs="?", default=json.dumps({
        'C': [25, 40], 
        'gamma': [0.0001, 0.0006],  
        'kernel': ['rbf']  
    }), help="JSON string for SVM parameter distribution")
    parser.add_argument("--fnn_momentum", nargs="?", default=0.7, help="Momentum for feed forward neural network", type=float)
    parser.add_argument("--fnn_weight_decay", nargs="?", default=0.01, help="Weight decay for feed forward neural network", type=float)
    parser.add_argument("--rnn_weight_decay", nargs="?", default=0.008, help="Weight decay for recurrent neural network", type=float)
    parser.add_argument("--fnn_learning_rate", nargs="?", default=0.075, help="Learning rate for feed forward neural network", type=float)
    parser.add_argument("--rnn_learning_rate", nargs="?", default=1e-3, help="Learning rate for recurrent neural network", type=float)
    parser.add_argument("--fnn_hidden_dim", nargs="?", default=20, help="Hidden dimension for feed forward neural network", type=int)
    parser.add_argument("--rnn_hidden_dim", nargs="?", default=10, help="Hidden dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_layer_dim", nargs="?", default=1, help="Layer dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_seq_dim", nargs="?", default=1, help="Sequence dimension for recurrent neural network", type=int)
    parser.add_argument("--output_dim", nargs="?", default=1, help="Output dimension for neural network", type=int)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--label_names", nargs="+", default=["age"], help="Predicted parameters, list")
    parser.add_argument("--column_to_copy", nargs="+", default=['male'], help="Columns to copy")
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female', 'weight', 'hight'], help="Columns to drop", type=list)
    parser.add_argument("--first_quantile", nargs="?", default=0.01, help="First quantile for svm regression", type=float)
    parser.add_argument("--last_quantile", nargs="?", default=0.99, help="Last quantile for svm regression", type=float)
    parser.add_argument("--plot", nargs="?", default=0, help="Plot results", type=bool)
    args = parser.parse_args()
    main(args)
