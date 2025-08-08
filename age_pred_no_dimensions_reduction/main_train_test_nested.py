import pandas as pd
import numpy as np
from torch import nn
import argparse
import joblib
import os
import torch
from sklearn.model_selection import train_test_split
from utils import prepare_dataset, train, valid, test, nn_data, dimensions_reduction
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    trainer = train.ModelTrainer()
    tester = test.ModelTester()
    reductor = dimensions_reduction.DimensionsReductor()

    mses, rmses, maes, count_outliers_lower, count_outliers_upper = [], [], [], [], []
    
    loss_fn = nn.L1Loss()
    global_config, model_config = preprocessor.load_model_config(args.model_name, args.config_file)

    model_path=f'models/{args.atlas}/{args.model_name}_{args.data_type}_valid_{args.valid}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    df = pd.read_csv(f'data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/all_concatenated.csv', sep='\t')
    

    is_big = 'big' in args.data_type
    id_file = Path(f"data/preprocessed_atlas/leave_out_identifiers{'_big' if is_big else ''}.csv")
    leave_out_csv = Path(f"data/preprocessed_atlas/{args.data_type}_norm_confirmed_{args.atlas}/leave_out{'_big' if is_big else ''}.csv")

    if id_file.exists():
        leave_ids = pd.read_csv(id_file)['identifier']
        df_leave = df[df['identifier'].isin(leave_ids)]
        df = df[~df['identifier'].isin(leave_ids)]
    else:
        df, df_leave = train_test_split(df, test_size=0.15, random_state=42)
        df_leave['identifier'].to_csv(id_file, index=False)
    df_leave.to_csv(leave_out_csv, sep='\t', index=False)


    identifier=df['identifier']
    df = df.drop(columns=args.columns_to_drop)
    input_dim = df.shape[1]-1

    if args.test_data_type!="None":
        df_test = pd.read_csv(f'data/{args.test_data_type}_norm_confirmed/all_concatenated.csv', sep='\t')
        identifier=df_test['identifier']
        df_test = df_test.drop(columns=args.columns_to_drop)

    X_trainval = df.drop(columns=args.label_names)
    X_val = None
    y_val = None
    y_trainval = df[args.label_names]
    i=0

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    all_selected_features = []

    for outer_train_idx, outer_val_idx in outer_cv.split(X_trainval, y_trainval):
        i+=1
        X_outer_train, X_outer_val = X_trainval.iloc[outer_train_idx], X_trainval.iloc[outer_val_idx]
        y_outer_train, y_outer_val = y_trainval.iloc[outer_train_idx], y_trainval.iloc[outer_val_idx]

        X_outer_train, X_val, X_outer_val, scaler = preprocessor.standardize_data(X_outer_train, X_val, X_outer_val, column_to_copy=args.column_to_copy)
        joblib.dump(scaler, f'{model_path}/scaler_train_nr_{i}.pkl')
        y_outer_val['identifier'] = identifier
        y_outer_val['male']=X_outer_val['male']

        feature=args.label_names

        print("Odchylenie",np.std(y_outer_train[feature[0]]))
        print("Srednia", np.mean(y_outer_train[feature[0]]))
        print("X_train", X_outer_train.shape)
        print("X_test", X_outer_val.shape)

        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        feature_scores = Counter()

        for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
            X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]
        

            if args.model_name=='forest':
                forest_param_dist = model_config.get("param_dist")
                forest_param_dist = preprocessor.convert_dist_params(forest_param_dist)
                rf = trainer.random_forrest_regression_model(X_inner_train, y_inner_train, forest_param_dist, *feature)
                mse, rmse, mae, results_df = tester.random_forest_regression_model(X_inner_val, y_inner_val, feature, rf)
                joblib.dump(rf, f'{model_path}/model_train_nr_{i}.pkl')


            elif args.model_name=="svm":
                svm_param_dist = model_config.get("param_dist")
                svm_param_dist = preprocessor.convert_dist_params(svm_param_dist)
                #features=['APARC-ctx-rh-transversetemporal_ThickStd', 'APARC-ctx-lh-superiorparietal_ThickStd', 'APARC-ctx-lh-caudalanteriorcingulate_ThickStd', 'APARC-ctx-rh-entorhinal_ThickStd', 'APARC-ctx-rh-inferiorparietal_ThickStd', 'APARC-ctx-lh-temporalpole_ThickStd', 'APARC-ctx-rh-pericalcarine_GrayVol', 'APARC-ctx-rh-caudalanteriorcingulate_ThickStd']
                clf = svm.SVR()
                model = RandomizedSearchCV(clf, param_distributions = svm_param_dist, n_iter=10, cv=5) 
                model.fit(X_inner_train, y_inner_train[feature].values.ravel())
                #importances = model.feature_importances_
                result = permutation_importance(
                model.best_estimator_,
                X_inner_train,
                y_inner_train,
                scoring='neg_mean_absolute_error',
                n_repeats=5,
                random_state=42
            )
                top_features = X_inner_train.columns[np.argsort(result.importances_mean)[-10:]]
                feature_scores.update(top_features)

            
            elif args.model_name=='fnn':
                y_inner_train[feature] = y_inner_train[feature]/100
                y_inner_val[feature] = y_inner_val[feature]/100
                train_dataloader = nn_data.load_fnn_data(X_inner_train, y_inner_train, model_config['batch_size'], feature)
                model = trainer.feed_forward_neural_network(train_dataloader, input_dim, model_config['hidden_dim'], model_config['output_dim'], model_config['learning_rate'], loss_fn, model_config['num_epochs'],  model_config['momentum'],  model_config['weight_decay'])
                mse, rmse, mae, results_df, feature_importance = tester.neural_network_regression(X_outer_val, y_outer_val, model_config['batch_size'], model, feature)
                torch.save(model.state_dict(), f'{model_path}/model_train_nr_{i}.pth')
                

            elif args.model_name=='rnn':
                y_inner_train[feature] = y_inner_train[feature]/100
                y_inner_val[feature] = y_inner_val[feature]/100
                train_dataloader = nn_data.load_rnn_data(X_inner_train, y_inner_train, model_config['batch_size'], feature)
                model = trainer.recurrent_neural_network(train_dataloader, model_config['seq_dim'], input_dim,  model_config['hidden_dim'], model_config['layer_dim'],  model_config['output_dim'], model_config['learning_rate'], loss_fn,  model_config['num_epochs'],  model_config['weight_decay'])
                mse, rmse, mae, results_df = tester.recurrent_neural_network_regression(X_outer_val, y_outer_val, model_config['batch_size'], model_config['seq_dim'], input_dim, model, feature)
                torch.save(model.state_dict(), f'{model_path}/model_train_nr_{i}.pth')

        if args.model_name=='svm':
            
            selected_features = [feat for feat, count in feature_scores.items() if count >= 2]
            print(f'Final features: {selected_features}')
            X_outer_train = X_outer_train[selected_features]
            X_outer_val = X_outer_val[selected_features]
            
            clf = trainer.svm_regression_model(X_outer_train, y_outer_train, svm_param_dist, feature)
            if args.valid:
                z, z_quantiles= valid.svm_regression_model(X_val, y_val, clf, feature, plot=args.plot)
            else:
                z=None
                z_quantiles=None
            mse, rmse, mae, results_df, feature_importance = tester.svm_regression_model(X_outer_val, y_outer_val, clf, z=z, feature=feature, comp = False, importance = True, shap_bool=args.shap)
            identifiers_lower, identifiers_upper, sex_lower, sex_upper = tester.svm_regression_model_quantiles(results_df, y_outer_val, z_quantiles=z_quantiles, feature=feature, plot=args.plot, first_quantile=args.first_quantile, last_quantile=args.last_quantile)
            joblib.dump(clf, f'{model_path}/model_train_nr_{i}.pkl')
            joblib.dump(z, f'{model_path}/z_train_nr_{i}.pkl')
            joblib.dump(z_quantiles, f'{model_path}/z_quantiles_train_nr_{i}.pkl')
            identifiers_lower = pd.Series(identifiers_lower, name=f'identifier_lower_{i}')
            sex_lower = pd.Series(sex_lower, name=f'male_lower_{i}')
            identifiers_upper = pd.Series(identifiers_upper, name=f'identifier_upper_{i}')
            sex_upper = pd.Series(sex_upper, name=f'male_upper_{i}')
            identifiers = pd.concat([identifiers_lower, sex_lower, identifiers_upper, sex_upper], axis=1)
            count_outliers_lower.append(len(identifiers_lower))
            count_outliers_upper.append(len(identifiers_upper))


            
            results_directory=f'{args.results_directory}/{args.atlas}'
            if not os.path.exists(results_directory):
                os.makedirs(results_directory)

            if i==0:
                results_df.to_csv(f'{results_directory}/train_{args.data_type}_test_{args.test_data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
                
                if args.model_name=='svm':
                    identifiers.to_csv(f'{results_directory}/train_{args.data_type}_test_{args.test_data_type}_identifiers_{args.model_name}_valid_{args.valid}.csv', sep='\t', index=False)
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

    count_outliers_lower = np.array([x if x is not None else np.nan for x in count_outliers_lower])
    count_outliers_upper = np.array([x if x is not None else np.nan for x in count_outliers_upper])
    print("Outliers lower", np.nanmean(count_outliers_lower)/y_outer_val.shape[0], np.nanstd(count_outliers_lower)/y_outer_val.shape[0])
    print("Outliers upper", np.nanmean(count_outliers_upper)/y_outer_val.shape[0], np.nanstd(count_outliers_upper)/y_outer_val.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser for age preidction - testing on test set without dimensionality reduction")
    parser.add_argument("--config_file", nargs="?", default="config/models.yaml", help="Configuration file", type=str)
    parser.add_argument("--atlas", nargs="?", default="a2009", help="atlas", type=str)
    parser.add_argument("--model_name", nargs="?", default="svm", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--valid", nargs="?", default=0, help="create valid set: 0/1", type=bool)
    parser.add_argument("--shap", nargs="?", default=0, help="calculate shap values", type=bool)
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--test_size", nargs="?", default=0.2, help="Size of test dataset", type=float)
    parser.add_argument("--test_data_type", nargs="?", default="None", help="Type of test dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--sex_subset", nargs="?", default="all", help="Choose the sex subset: all/female/male", type=str)
    parser.add_argument("--division_by_total_volume", nargs="?", default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0", type=bool)
    parser.add_argument("--n_most_important_features", nargs="?", default=20, help="Choose the number of extracting features that load into components")
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--column_to_copy", nargs="?", default=['male'], help="Columns to copy", type=list)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female', 'weight', 'hight'], help="Columns to drop", type=list)
    parser.add_argument("--first_quantile", nargs="?", default=0.01, help="First quantile for svm regression", type=float)
    parser.add_argument("--last_quantile", nargs="?", default=0.99, help="Last quantile for svm regression", type=float)
    parser.add_argument("--plot", nargs="?", default=0, help="Plot results", type=bool)
    args = parser.parse_args()
    main(args)
