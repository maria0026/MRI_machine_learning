import pandas as pd
import numpy as np
from torch import nn
import argparse
from utils import dimensions_reduction, prepare_dataset, plots, train, test

def main(args):

    df=pd.read_csv(f'data/{args.data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    df=df.drop(columns=args.columns_to_drop)

    df_test=pd.read_csv(f'data/{args.test_data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    df_test=df_test.drop(columns=args.columns_to_drop)

    column_to_copy='male'
    mses, rmses, maes =[], [], []
    input_dim = args.components_nr
    loss_fn = nn.MSELoss()


    for i in range(args.n_crosval):
        if args.division_by_total_volume:
            prepare_dataset.divide_by_total_volume(df)
            prepare_dataset.divide_by_total_volume(df_test)

        X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, args.label_names)
        X_train_to_stardarize=X_train.drop(columns=column_to_copy)
        X_test_to_stardarize=X_test.drop(columns=column_to_copy)
        X_train_standarized, X_test_stantarized=prepare_dataset.standarize_data(X_train_to_stardarize, X_test_to_stardarize)
        X_train=pd.concat([X_train_standarized, X_train[column_to_copy]], axis=1)
        X_test=pd.concat([X_test_stantarized, X_test[column_to_copy]], axis=1)

        #PCA
        pca_mri, train_pca, test_pca, importance_df=dimensions_reduction.principal_component_analysis(X_train, X_test, args.components_nr, args.n_most_important_features)
        explained_variance_ratio=pca_mri.explained_variance_ratio_
        formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
        print('Explained variability per principal component: {}'.format(formatted_explained_variance))

        train_principal_Df = pd.DataFrame(data = train_pca
                    , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

        test_principal_Df = pd.DataFrame(data = test_pca
                    , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)

        if args.sex_subset=='all':
            X_train=train_principal_Df
            X_test=test_principal_Df 

        elif args.sex_subset=='male':
            train_indices=(X_train[X_train['male']==1]).index.tolist()
            test_indices=(X_test[X_test['male']==1]).index.tolist()

            X_train=train_principal_Df.loc[train_indices]
            y_train=y_train.loc[train_indices]
            X_test=test_principal_Df.loc[test_indices]
            y_test=y_test.loc[test_indices]


        elif args.sex_subset=='female':
            train_indices=(X_train[X_train['male']==0]).index.tolist()
            test_indices=(X_test[X_test['male']==0]).index.tolist()

            X_train=train_principal_Df.loc[train_indices]
            y_train=y_train.loc[train_indices]
            X_test=test_principal_Df.loc[test_indices]
            y_test=y_test.loc[test_indices]
  

        feature=args.label_names
        print("Odchylenie",np.std(y_train[feature[0]]))
        print("Srednia", np.mean(y_train[feature[0]]))
        

        if args.model_name=='forest':
            rf=train.random_forrest_regression_model(X_train, y_train, *feature)
            best_rf = rf.best_estimator_
            feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            feature_importances.index = feature_importances.index.astype(int)
            print(feature_importances)
            #sort ascending by indexes
            feature_importances=feature_importances.sort_index()
            print(feature_importances.index)
            print("importances", feature_importances.values)
            importance_df['comp_imp']=feature_importances.values
            mse, rmse, mae, results_df= test.random_forest_regression_model(X_test, y_test, feature, rf)


        elif args.model_name=="svm":
            clf=train.svm_regression_model(X_train, y_train)
            mse, rmse, mae, results_df=test.svm_regression_model(X_test, y_test, clf)
            

        elif args.model_name=='fnn':
            y_train=y_train/100
            y_test=y_test/100
            model=train.layer_neural_network(X_train, y_train, input_dim, args.fnn_hidden_dim, args.output_dim, args.fnn_learning_rate, loss_fn, args.num_epochs, args.fnn_momentum, args.fnn_weight_decay)
            mse, rmse, mae, results_df, df_fi=test.neural_network_regression(X_test, y_test, model)
    

        elif args.model_name=='rnn':
            y_train=y_train/100
            y_test=y_test/100
            model=train.recurrent_neural_network(X_train, y_train, args.rnn_seq_dim, input_dim, args.rnn_hidden_dim, args.rnn_layer_dim, args.output_dim, args.rnn_learning_rate, loss_fn, args.num_epochs, args.rnn_weight_decay)
            mse, rmse, mae, results_df=test.recurrent_neural_network_regression(X_test, y_test, args.rnn_seq_dim, input_dim, model)
        

        if args.model_name!='nn':
            if i==0:
                importance_df.to_csv(f'{args.results_directory}/importance_age_{args.model_name}.csv', sep='\t')
                results_df.to_csv(f'{args.results_directory}/regression_results_{args.model_name}.csv', sep='\t')
            else:
                importance_df_old=pd.read_csv(f'{args.results_directory}/importance_age_{args.model_name}.csv', sep='\t', index_col=0)
                importance_df=pd.concat([importance_df_old, importance_df], axis=1)
                importance_df.to_csv(f'{args.results_directory}/importance_age_{args.model_name}.csv', sep='\t', index=True)

                results_df_old=pd.read_csv(f'{args.results_directory}/regression_results_{args.model_name}.csv', sep='\t')
                results_df=pd.concat([results_df_old.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
                results_df.to_csv(f'{args.results_directory}/regression_results_{args.model_name}.csv', sep='\t', index=False)

        else:
            if i==0:
                df_fi=pd.concat([df_fi.reset_index(drop=True), importance_df.reset_index(drop=True)], axis=1)
                df_fi.to_csv(f'{args.results_directory}/importance_age_{args.model_name}.csv', sep='\t', index=False)

                results_df.to_csv(f'{args.results_directory}/regression_results_{args.model_name}.csv', sep='\t', index=False)
            else:
                importance_df_old=pd.read_csv(f'{args.results_directory}/importance_age_{args.model_name}.csv', sep='\t')
                importance_df=pd.concat([importance_df_old.reset_index(drop=True), df_fi.reset_index(drop=True), importance_df.reset_index(drop=True),], axis=1)
                importance_df.to_csv(f'{args.results_directory}/importance_age_{args.model_name}.csv', sep='\t', index=False)
                    
                results_df_old=pd.read_csv(f'{args.results_directory}/regression_results_{args.model_name}.csv', sep='\t')
                results_df=pd.concat([results_df_old.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
                results_df.to_csv(f'{args.results_directory}/regression_results_{args.model_name}.csv', sep='\t', index=False)
        
        
        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)


    print("Mean squared error", np.mean(mses), np.std(mses))
    print("Root mean squared error", np.mean(rmses), np.std(rmses))
    print("Mean absolute error", np.mean(maes), np.std(maes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for age preidction")
    parser.add_argument("--model_name", nargs="?", default="rnn", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--test_size", nargs="?", default=0.2, help="Size of test dataset", type=float)
    parser.add_argument("--test_data_type", nargs="?", default="positive", help="Type of test dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--sex_subset", nargs="?", default="all", help="Choose the sex subset: all/female/male", type=str)
    parser.add_argument("--division_by_total_volume", nargs="?", default=True, help="Divide volumetric data by Estimated_Total_Intracranial_Volume", type=bool)
    parser.add_argument("--n_most_important_features", nargs="?", default=20, help="Choose the number of extracting features that load into components")
    parser.add_argument("--components_nr", nargs="?", default=35, help="Number of components for principal component analysis", type=int)
    parser.add_argument("--n_crosval", nargs="?", default=5, help="Number of crossvalidation", type=int)
    parser.add_argument("--num_epochs", nargs="?", default=100, help="Number of epochs", type=int)
    parser.add_argument("--fnn_momentum", nargs="?", default=0.7, help="Momentum for feed forward neural network", type=float)
    parser.add_argument("--_fnn_weight_decay", nargs="?", default=0.01, help="Weight decay for feed forward neural network", type=float)
    parser.add_argument("--rnn_weight_decay", nargs="?", default=0.008, help="Weight decay for recurrent neural network", type=float)
    parser.add_argument("--fnn_learning_rate", nargs="?", default=0.075, help="Learning rate for feed forward neural network", type=float)
    parser.add_argument("--rnn_learning_rate", nargs="?", default=1e-3, help="Learning rate for recurrent neural network", type=float)
    parser.add_argument("--fnn_hidden_dim", nargs="?", default=20, help="Hidden dimension for feed forward neural network", type=int)
    parser.add_argument("--rnn_hidden_dim", nargs="?", default=10, help="Hidden dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_layer_dim", nargs="?", default=1, help="Layer dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_seq_dim", nargs="?", default=1, help="Sequence dimension for recurrent neural network", type=int)
    parser.add_argument("--output_dim", nargs="?", default=1, help="Output dimension for neural network", type=int)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier', 'norm_confirmed', 'sex', 'female'], help="Columns to drop", type=list)
    parser.add_argument("--different_test_dataset", nargs="?", default=False, help="Use different test dataset", type=bool)
    args = parser.parse_args()
    main(args)
