import pandas as pd
import numpy as np
from torch import nn
import argparse
from scipy.stats import randint, uniform
import json
import joblib
import os
import torch
import yaml
from sklearn.model_selection import train_test_split
from utils import prepare_dataset, train, valid, test, nn_data, dimensions_reduction



def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    trainer = train.ModelTrainer()
    tester = test.ModelTester()
    reductor = dimensions_reduction.DimensionsReductor()

    mses, rmses, maes = [], [], []
    loss_fn = nn.MSELoss()

    global_config, model_config = preprocessor.load_model_config(args.model_name, args.config_file)

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


    for i in range(args.n_crosval):

        #splitting and standardizing
        feature=args.label_names
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(df, args.label_names, test_size=global_config['test_size'], valid=args.valid)
        X_train, X_val, X_test, scaler = preprocessor.standardize_data(X_train, X_val, X_test, column_to_copy=args.column_to_copy)
        joblib.dump(scaler, f'{model_path}/scaler_train_nr_{i}.pkl')
        y_test['identifier'] = identifier
        y_test['male']=X_test['male']
        print("Odchylenie",np.std(y_train[feature[0]]))
        print("Srednia", np.mean(y_train[feature[0]]))
        print("X_train", X_train.shape)
        print("X_test", X_test.shape)

        if args.model_name=='forest':
            forest_param_dist = model_config.get("param_dist")
            forest_param_dist = preprocessor.convert_dist_params(forest_param_dist)
            rf = trainer.random_forrest_regression_model(X_train, y_train, forest_param_dist, *feature)
            mse, rmse, mae, results_df = tester.random_forest_regression_model(X_test, y_test, feature, rf)
            joblib.dump(rf, f'{model_path}/model_train_nr_{i}.pkl')


        elif args.model_name=="svm":
            svm_param_dist = model_config.get("param_dist")
            svm_param_dist = preprocessor.convert_dist_params(svm_param_dist)

            y_train_h=y_train.copy()
            y_train_h['identifier'] = identifier
            y_train_h['male']=X_train['male']
            rng = np.random.default_rng(42)  # for reproducibility
            #selected_columns = rng.choice(X_train.columns, size=150, replace=False)

            # Reduce X_train to n features
            #X_train_h = X_train[selected_columns]
            X_train_h = X_train
        
            '''
            features=reductor.hierarchical_feature_selection(X_train_h, y_train_h, trainer, tester, args.model_name, svm_param_dist, feature)
            print("Selected features", features)
            X_train_selected = X_train[features]
            X_val_selected = X_val[features]
            X_test_selected = X_test[features]
            '''
            X_train_selected = X_train
            X_val_selected = X_val
            X_test_selected = X_test

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
            train_dataloader = nn_data.load_fnn_data(X_train, y_train,  model_config['batch_size'], feature)
            model = trainer.feed_forward_neural_network(train_dataloader, input_dim, model_config['hidden_dim'], model_config['output_dim'], model_config['learning_rate'], loss_fn, model_config['num_epochs'],  model_config['momentum'],  model_config['weight_decay'])
            mse, rmse, mae, results_df, feature_importance = tester.neural_network_regression(X_test, y_test,  model_config['batch_size'], model,feature)
            torch.save(model.state_dict(), f'{model_path}/hierarchical_model_train_nr_{i}.pth')
            

        elif args.model_name=='rnn':
            y_train[feature] = y_train[feature]/100
            y_test[feature] = y_test[feature]/100
            train_dataloader = nn_data.load_rnn_data(X_train, y_train, model_config['batch_size'], feature)
            model = trainer.recurrent_neural_network(train_dataloader,  model_config['seq_dim'], input_dim,  model_config['hidden_dim'],  model_config['layer_dim'],  model_config['output_dim'], model_config['learning_rate'], loss_fn,  model_config['num_epochs'],  model_config['weight_decay'])
            mse, rmse, mae, results_df = tester.recurrent_neural_network_regression(X_test, y_test, model_config['batch_size'],  model_config['seq_dim'], input_dim, model, feature)
            torch.save(model.state_dict(), f'{model_path}/hierarchical_model_train_nr_{i}.pth')

        results_directory=f'{args.results_directory}/{args.atlas}_hierarchical'
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        if i==0:
            results_df.to_csv(f'{results_directory}/hierarchical_train_{args.data_type}_test_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
        else:
            results_df_old = pd.read_csv(f'{results_directory}/hierarchical_train_{args.data_type}_test_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t')
            results_df = pd.concat([results_df_old.reset_index(drop = True), results_df.reset_index(drop = True)], axis = 1)
            results_df.to_csv(f'{results_directory}/hierarchical_train_{args.data_type}_test_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)

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
    parser = argparse.ArgumentParser("Parser for age prediction - testing on test set with hierachical clustering feature selection")
    parser.add_argument("--config_file", nargs="?", default="config/models.yaml", help="Configuration file", type=str)
    parser.add_argument("--atlas", nargs="?", default="ASEG", help="atlas", type=str)
    parser.add_argument("--model_name", nargs="?", default="forest", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--valid", nargs="?", default=1, help="create valid set: 0/1", type=bool)
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--sex_subset", nargs="?", default="all", help="Choose the sex subset: all/female/male", type=str)
    parser.add_argument("--n_most_important_features", nargs="?", default=20, help="Choose the number of extracting features that load into components")
    parser.add_argument("--n_crosval", nargs="?", default=5, help="Number of crossvalidation", type=int)
    parser.add_argument("--batch_size", nargs="?", default=64, help="Batch size", type=int)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--label_names", nargs="+", default=["age"], help="Predicted parameters")
    parser.add_argument("--column_to_copy", nargs="+", default=['male'], help="Columns to copy")
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female', 'weight', 'hight'], help="Columns to drop", type=list)
    parser.add_argument("--plot", nargs="?", default=0, help="Plot results", type=bool)
    args = parser.parse_args()
    main(args)
