import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import dimensions_reduction
import prepare_dataset
import plots
import train
import test
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

n_crosval=5
tprs=[]
fprs=[]
aucs=[]
accuracies=[]
precisions=[]
recalls=[]
cms=[]
label_names=['male']

df=pd.read_csv('data_4/normal/all_concatenated.csv', sep='\t')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])

df_test=pd.read_csv('data_4/test_data_normal/all_concatenated.csv', sep='\t')
df_test=df_test.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])

#df=pd.read_csv('data_4/all/all_concatenated.csv', sep='\t')
#df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])

#drop columns containing norm and Std
#df=df.drop(columns=[col for col in df.columns if 'norm' in col or 'Std' in col])

for i in range(n_crosval):
    prepare_dataset.divide_by_total_volume(df)
    X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, label_names)
    X_test=df_test.drop(columns=label_names)
    y_test=df_test[label_names]
    X_train, X_test=prepare_dataset.standarize_data(X_train, X_test)

    #PCA
    components_nr=35
    pca_mri, train_pca, test_pca, importance_df=dimensions_reduction.principal_component_analysis(X_train, X_test, components_nr)
    
    explained_variance_ratio=pca_mri.explained_variance_ratio_
    formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
    print('Explained variability per principal component: {}'.format(formatted_explained_variance))
    print(len(formatted_explained_variance))
    print(train_pca.shape, test_pca.shape)
    #plots
    #plots.pca(pca_mri, train_pca, test_pca, X_train, y_train, X_test, y_test)
    #plots.scree_plot(pca_mri)

    #mamy teraz train_pca i test_pca
    train_principal_Df = pd.DataFrame(data = train_pca
                , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

    test_principal_Df = pd.DataFrame(data = test_pca
                , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)

    #train_principal_Df['age']=X_train['age']
    #test_principal_Df['age']=X_test['age']


    X_train=train_principal_Df
    X_test=test_principal_Df 
    feature='male'

    '''
    #svm
    clf=train.svm_classification_model(X_train, y_train)
    accuracy, precision, recall, cm, fpr, tpr=test.svm_classification_model(X_test, y_test, clf)
    '''

    '''
    #random forest    
    rf=train.random_forest_model(X_train, y_train, feature)
    best_rf = rf.best_estimator_
    feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    feature_importances.index = feature_importances.index.astype(int)
    print(feature_importances)
    #sort ascending by indexes
    feature_importances=feature_importances.sort_index()
    print(feature_importances.index)
    print("wart", feature_importances.values)
    importance_df['component_importance_on_tree']=feature_importances.values
    #save to csv
    if i==0:
        importance_df.to_csv('importance_sex.csv', sep='\t', index=True)
    else:
        #concatenate
        importance_df_old=pd.read_csv('importance_sex.csv', sep='\t', index_col=0)
        importance_df=pd.concat([importance_df_old, importance_df], axis=1)
        importance_df.to_csv('importance_sex.csv', sep='\t', index=True)

    accuracy, precision, recall, cm, fpr, tpr=test.random_forest_model(X_test, y_test, feature, rf)
    '''
    

    
    #neural network
    input_dim = components_nr
    hidden_dim = 10
    output_dim = 1
    learning_rate = 0.075
    loss_fn = nn.BCELoss()
    num_epochs = 100
    model=train.neural_network_classification(X_train, y_train, input_dim, hidden_dim, output_dim, learning_rate, loss_fn, num_epochs)
    accuracy, precision, recall, FPR, cm, fpr, tpr, df_fi=test.neural_network_classification(X_test, y_test, model)
    if i==0:
        df_fi=pd.concat([df_fi.reset_index(drop=True), importance_df.reset_index(drop=True)], axis=1)
        df_fi.to_csv('importance_sex_nn.csv', sep='\t', index=False)
    else:
        #concatenate
        importance_df_old=pd.read_csv('importance_sex_nn.csv', sep='\t', index_col=0)
        importance_df=pd.concat([importance_df_old.reset_index(drop=True), importance_df.reset_index(drop=True), df_fi.reset_index(drop=True)], axis=1)
        importance_df.to_csv('importance_sex_nn.csv', sep='\t', index=False)
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    tprs.append(tpr)
    fprs.append(fpr)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plots.confusion_matrix(cm)
    #plots.random_forest_features(X_train, rf)
    #plots.roc_curve(fpr, tpr)

print("Accuracy", np.mean(accuracies), np.std(accuracies))
print("Precision", np.mean(precisions), np.std(precisions))
print("Recall", np.mean(recalls), np.std(recalls))
tprs = np.array([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(tpr)), tpr) for tpr in tprs])
fprs = np.array([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(fpr)), fpr) for fpr in fprs])
print("auc", np.mean(aucs), np.std(aucs)) 
print(aucs)
plots.roc_curve_crossvalid(tprs, fprs, aucs, feature)
