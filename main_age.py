import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch import nn
from torch.utils.data import Dataset, DataLoader


from utils import dimensions_reduction
from utils import prepare_dataset
from utils import plots
from utils import train
from utils import test


model_name="nn"
components_nr=40

df=pd.read_csv('data/positive_norm_confirmed_normal/all_concatenated.csv', sep='\t')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])
label_names=['age']
column_to_copy='male'
n_crosval=5
n_most_important_features=5
mses=[]
rmses=[]
maes=[]
results_directory='results'
input_dim = components_nr
hidden_dim = 20
layer_dim = 1
output_dim = 1
seq_dim=1

loss_fn = nn.MSELoss()
num_epochs = 100


for i in range(n_crosval):
    prepare_dataset.divide_by_total_volume(df)
    X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, label_names)
    X_train_to_stardarize=X_train.drop(columns=column_to_copy)
    X_test_to_stardarize=X_test.drop(columns=column_to_copy)
    X_train_standarized, X_test_stantarized=prepare_dataset.standarize_data(X_train_to_stardarize, X_test_to_stardarize)
    X_train=pd.concat([X_train_standarized, X_train[column_to_copy]], axis=1)
    X_test=pd.concat([X_test_stantarized, X_test[column_to_copy]], axis=1)

    #PCA
    pca_mri, train_pca, test_pca, importance_df=dimensions_reduction.principal_component_analysis(X_train, X_test, components_nr, n_most_important_features)
    explained_variance_ratio=pca_mri.explained_variance_ratio_
    formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
    print('Explained variability per principal component: {}'.format(formatted_explained_variance))

    train_principal_Df = pd.DataFrame(data = train_pca
                , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

    test_principal_Df = pd.DataFrame(data = test_pca
                , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)

    column_to_copy='male'

    '''
    #2 osobne modele- dla mężczyzn i kobiet

    X_train_male=train_principal_Df[train_principal_Df['male']==1]
    train_indices=X_train_male.index.tolist()
    y_train_male=y_train.loc[train_indices]

    X_test_male=test_principal_Df[test_principal_Df['male']==1] 
    test_indices=X_test_male.index.tolist()
    y_test_male=y_test.loc[test_indices]

    #female
    X_train_female=train_principal_Df[train_principal_Df['male']==0]
    train_indices=X_train_female.index.tolist()
    y_train_female=y_train.loc[train_indices]

    X_test_female=test_principal_Df[test_principal_Df['male']==0]
    test_indices=X_test_female.index.tolist()
    y_test_female=y_test.loc[test_indices]
    '''

    X_train=train_principal_Df
    X_test=test_principal_Df 

    '''
    X_train=X_train_male
    y_train=y_train_male
    X_test=X_test_male
    y_test=y_test_male


    X_train=X_train_female
    y_train=y_train_female
    X_test=X_test_female
    y_test=y_test_female
    '''

    feature='age'
    print("Odchylenie",np.std(y_train[feature]))
    print("Srednia", np.mean(y_train[feature]))
    

    if model_name=='forest':
        rf=train.random_forrest_regression_model(X_train, y_train, feature)
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


    elif model_name=="svm":
        clf=train.svm_regression_model(X_train, y_train)
        mse, rmse, mae, results_df=test.svm_regression_model(X_test, y_test, clf)
        

    elif model_name=='nn':
        momentum=0.7
        weight_decay=0.01
        learning_rate = 0.075
        y_train=y_train/100
        y_test=y_test/100
        model=train.layer_neural_network(X_train, y_train, input_dim, hidden_dim, output_dim, learning_rate, loss_fn, num_epochs, momentum, weight_decay)
        mse, rmse, mae, results_df, df_fi=test.neural_network_regression(X_test, y_test, model)
 

    elif model_name=='rnn':
        weight_decay=0.008
        learning_rate=1e-3
        y_train=y_train/100
        y_test=y_test/100
        model=train.recurrent_neural_network(X_train, y_train, seq_dim, input_dim, hidden_dim, layer_dim, output_dim, learning_rate, loss_fn, num_epochs, weight_decay)
        mse, rmse, mae, results_df=test.recurrent_neural_network_regression(X_test, y_test, seq_dim, input_dim, model)
    

    if model_name!='nn':
        if i==0:
            importance_df.to_csv(f'{results_directory}/importance_age_{model_name}.csv', sep='\t')

            results_df.to_csv(f'{results_directory}/regression_results_{model_name}.csv', sep='\t')
        else:
            #concatenate
            importance_df_old=pd.read_csv(f'{results_directory}/importance_age_{model_name}.csv', sep='\t', index_col=0)
            importance_df=pd.concat([importance_df_old, importance_df], axis=1)
            importance_df.to_csv(f'{results_directory}/importance_age_{model_name}.csv', sep='\t', index=True)

            results_df_old=pd.read_csv(f'{results_directory}/regression_results_{model_name}.csv', sep='\t')
            results_df=pd.concat([results_df_old.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
            results_df.to_csv(f'{results_directory}/regression_results_{model_name}.csv', sep='\t', index=False)

    else:
        if i==0:
            df_fi=pd.concat([df_fi.reset_index(drop=True), importance_df.reset_index(drop=True)], axis=1)
            df_fi.to_csv(f'{results_directory}/importance_age_{model_name}.csv', sep='\t', index=False)

            results_df.to_csv(f'{results_directory}/regression_results_{model_name}.csv', sep='\t', index=False)
        else:
            importance_df_old=pd.read_csv(f'{results_directory}/importance_age_{model_name}.csv', sep='\t')
            importance_df=pd.concat([importance_df_old.reset_index(drop=True), df_fi.reset_index(drop=True), importance_df.reset_index(drop=True),], axis=1)
            importance_df.to_csv(f'{results_directory}/importance_age_{model_name}.csv', sep='\t', index=False)
                
            results_df_old=pd.read_csv(f'{results_directory}/regression_results_{model_name}.csv', sep='\t')
            results_df=pd.concat([results_df_old.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
            results_df.to_csv(f'{results_directory}/regression_results_{model_name}.csv', sep='\t', index=False)
    
    
    mses.append(mse)
    rmses.append(rmse)
    maes.append(mae)


print("Mean squared error", np.mean(mses), np.std(mses))
print("Root mean squared error", np.mean(rmses), np.std(rmses))
print("Mean absolute error", np.mean(maes), np.std(maes))

