import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch import nn
import argparse
from scipy.stats import randint, uniform
from utils import dimensions_reduction, prepare_dataset, plots, train, valid, test, nn_data
import json

def main(args):

    preprocessor = prepare_dataset.DatasetPreprocessor()
    reductor = dimensions_reduction.DimensionsReductor()
    trainer = train.ModelTrainer()

    tprs, fprs, aucs, accuracies, precisions, recalls= [], [], [], [], [], []
    
    df = pd.read_csv(f'data/{args.data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    df = df.drop(columns=args.columns_to_drop)

    df_test = pd.read_csv(f'data/{args.test_data_type}_norm_confirmed_normal/all_concatenated.csv', sep='\t')
    df_test = df_test.drop(columns=args.columns_to_drop)

    #drop columns containing norm and Std
    #df=df.drop(columns=[col for col in df.columns if 'norm' in col or 'Std' in col])

    for i in range(args.n_crosval):
        if args.division_by_total_volume:
            df = preprocessor.divide_by_total_volume(df)
            df_test = preprocessor.divide_by_total_volume(df_test)

        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(df, args.label_names, args.test_size, valid=args.validation)
        if args.different_test_dataset:
            X_test = df_test.drop(columns=args.label_names)
            y_test = df_test[args.label_names]

        X_train, _,  X_test, _ = preprocessor.standardize_data(X_train, X_val, X_test)

        #PCA
        pca_mri, train_pca, _, test_pca, importance_df = reductor.principal_component_analysis(X_train, X_test, args.components_nr, n_features=args.n_most_important_features)
        explained_variance_ratio = pca_mri.explained_variance_ratio_
        formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
        print('Explained variability per principal component: {}'.format(formatted_explained_variance))
        print(train_pca.shape, test_pca.shape)

        train_principal_Df = pd.DataFrame(data = train_pca
                    , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

        test_principal_Df = pd.DataFrame(data = test_pca
                    , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)


        X_train = train_principal_Df
        X_test = test_principal_Df 
        feature = args.label_names

        input_dim = args.components_nr
        loss_fn = nn.BCELoss()

        if args.model_name=="forest":
            forest_param_dist = json.loads(args.forest_param_dist)
            forest_param_dist['n_estimators'] = randint(*forest_param_dist['n_estimators'])
            forest_param_dist['max_depth'] = randint(*forest_param_dist['max_depth'])
            forest_param_dist['min_samples_leaf'] = randint(*forest_param_dist['min_samples_leaf'])
            rf = trainer.random_forest_model(X_train, y_train, args.forest_param_dist, *feature)
            accuracy, precision, recall, cm, fpr, tpr=test.random_forest_model(X_test, y_test, feature, rf)
            best_rf = rf.best_estimator_
            feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            feature_importances.index = feature_importances.index.astype(int)
            #print(feature_importances)
            #sort ascending by indexes
            feature_importances = feature_importances.sort_index()

            importance_df['component_names'] = feature_importances.index
            importance_df['comp_imp'] = feature_importances.values


        elif args.model_name=="svm":
            svm_param_dist = json.loads(args.svm_param_dist)
            svm_param_dist['C'] = randint(*svm_param_dist['C'])  
            svm_param_dist['gamma'] = uniform(*svm_param_dist['gamma']) 
            clf = trainer.svm_classification_model(X_train, y_train, svm_param_dist)
            accuracy, precision, recall, cm, fpr, tpr = test.svm_classification_model(X_test, y_test, clf)


        elif args.model_name=="fnn":
            train_dataloader = nn_data.load_fnn_data(X_train, y_train, args.batch_size, feature)
            model = trainer.feed_forward_neural_network(train_dataloader, input_dim, args.fnn_hidden_dim, args.output_dim, args.fnn_learning_rate, loss_fn, args.num_epochs, args.fnn_momentum, args.fnn_weight_decay)
            accuracy, precision, recall, FPR, cm, fpr, tpr, df_fi = test.neural_network_classification(X_test, y_test, args.batch_size, model, feature)
            importance_df = pd.concat([df_fi.reset_index(drop=True), importance_df.reset_index(drop=True)], axis=1)

        elif args.model_name=="rnn":
            train_dataloader = nn_data.load_rnn_data(X_train, y_train, args.batch_size, feature)
            model = trainer.recurrent_neural_network(train_dataloader, args.rnn_seq_dim, input_dim, args.rnn_hidden_dim, args.rnn_layer_dim, args.output_dim, args.rnn_learning_rate, loss_fn, args.num_epochs, args.rnn_weight_decay)
            accuracy, precision, recall, FPR, cm, fpr, tpr = test.recurrent_neural_network_classification(X_test, y_test, args.batch_size, args.rnn_seq_dim, input_dim, model, feature)


        if i==0:
            importance_df.to_csv(f'{args.results_directory}/{args.data_type}_importance_sex_{args.model_name}.csv', sep='\t')
        else:
            importance_df_old = pd.read_csv(f'{args.results_directory}/{args.data_type}_importance_sex_{args.model_name}.csv', sep='\t', index_col=0)
            importance_df = pd.concat([importance_df_old, importance_df], axis=1)
            importance_df.to_csv(f'{args.results_directory}/{args.data_type}_importance_sex_{args.model_name}.csv', sep='\t', index=True)
      

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        tprs.append(tpr)
        fprs.append(fpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)


    print("Accuracy", np.mean(accuracies), np.std(accuracies))
    print("Precision", np.mean(precisions), np.std(precisions))
    print("Recall", np.mean(recalls), np.std(recalls))
    tprs = np.array([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(tpr)), tpr) for tpr in tprs])
    fprs = np.array([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(fpr)), fpr) for fpr in fprs])
    print("auc", np.mean(aucs), np.std(aucs)) 
    print(aucs)
    plots.roc_curve_crossvalid(tprs, fprs, aucs, feature)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for sex preidction")
    parser.add_argument("--model_name", nargs="?", default="svm", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--test_size", nargs="?", default=0.2, help="Size of test dataset", type=float)
    parser.add_argument("--test_data_type", nargs="?", default="positive", help="Type of test dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--validation", nargs="?", default=0, help="create validation set: 0/1", type=int)
    parser.add_argument("--division_by_total_volume", nargs="?", default=False, help="Divide volumetric data by Estimated_Total_Intracranial_Volume", type=bool)
    parser.add_argument("--n_most_important_features", nargs="?", default=20, help="Choose the number of extracting features that load into components")
    parser.add_argument("--components_nr", nargs="?", default=35, help="Number of components for principal component analysis", type=int)
    parser.add_argument("--n_crosval", nargs="?", default=5, help="Number of crossvalidation", type=int)
    parser.add_argument("--num_epochs", nargs="?", default=100, help="Number of epochs", type=int)
    parser.add_argument("--forest_param_dist", nargs="?", default=json.dumps({
        'n_estimators': [50, 500],
        'max_depth': [1, 35],
        'min_samples_leaf': [1, 8],
        'min_samples_split': [2, 3, 5, 8, 10, 14, 20]}),
        help="JSON string for random forest parameter distribution")
    parser.add_argument("--svm_param_dist", nargs="?", default=json.dumps({
        'C': [30, 40], 
        'gamma': [0.0001, 0.0005],  
        'kernel': ['rbf']  
    }), help="JSON string for SVM parameter distribution")
    parser.add_argument("--batch_size", nargs="?", default=64, help="Batch size", type=int)
    parser.add_argument("--fnn_momentum", nargs="?", default=0.3, help="Momentum for feed forward neural network", type=float)
    parser.add_argument("--fnn_weight_decay", nargs="?", default=0.08, help="Weight decay for feed forward neural network", type=float)
    parser.add_argument("--rnn_weight_decay", nargs="?", default=0.008, help="Weight decay for recurrent neural network", type=float)
    parser.add_argument("--fnn_learning_rate", nargs="?", default=0.075, help="Learning rate for feed forward neural network", type=float)
    parser.add_argument("--rnn_learning_rate", nargs="?", default=1e-3, help="Learning rate for recurrent neural network", type=float)
    parser.add_argument("--fnn_hidden_dim", nargs="?", default=20, help="Hidden dimension for feed forward neural network", type=int)
    parser.add_argument("--rnn_hidden_dim", nargs="?", default=10, help="Hidden dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_layer_dim", nargs="?", default=1, help="Layer dimension for recurrent neural network", type=int)
    parser.add_argument("--rnn_seq_dim", nargs="?", default=1, help="Sequence dimension for recurrent neural network", type=int)
    parser.add_argument("--output_dim", nargs="?", default=1, help="Output dimension for neural network", type=int)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    parser.add_argument("--label_names", nargs="?", default=["male"], help="Predicted parameters, list", type=list)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier', 'norm_confirmed', 'sex', 'female'], help="Columns to drop", type=list)
    parser.add_argument("--different_test_dataset", nargs="?", default=False, help="Use different test dataset", type=bool)
    args = parser.parse_args()
    main(args)
