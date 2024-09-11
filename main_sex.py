import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader
from torch import nn

from utils import dimensions_reduction
from utils import prepare_dataset
from utils import plots
from utils import train
from utils import test

model_name="rnn" 
components_nr=35

n_crosval=5
n_most_important_features=5
tprs=[]
fprs=[]
aucs=[]
accuracies=[]
precisions=[]
recalls=[]
cms=[]
label_names=['male']
results_directory='results'

df=pd.read_csv('data/positive_norm_confirmed_normal/all_concatenated.csv', sep='\t')

df_test=pd.read_csv('data/negative_norm_confirmed_normal/all_concatenated.csv', sep='\t')
df_test=df_test.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])

#df=pd.read_csv('data/all_norm_confirmed_normal/all_concatenated.csv', sep='\t')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])

#drop columns containing norm and Std
#df=df.drop(columns=[col for col in df.columns if 'norm' in col or 'Std' in col])

for i in range(n_crosval):
    #prepare_dataset.divide_by_total_volume(df)
    #prepare_dataset.divide_by_total_volume(df_test)
    X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, label_names)
    #X_test=df_test.drop(columns=label_names)
    #y_test=df_test[label_names]
    X_train, X_test=prepare_dataset.standarize_data(X_train, X_test)

    #PCA

    pca_mri, train_pca, test_pca, importance_df=dimensions_reduction.principal_component_analysis(X_train, X_test, components_nr, n_features=n_most_important_features)
    explained_variance_ratio=pca_mri.explained_variance_ratio_
    formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
    print('Explained variability per principal component: {}'.format(formatted_explained_variance))
    print(len(formatted_explained_variance))
    print(train_pca.shape, test_pca.shape)

    #plots.distribution(train_pca, test_pca)
    
    #plots
    #plots.pca(pca_mri, train_pca, test_pca, X_train, y_train, X_test, y_test)
    #plots.scree_plot(pca_mri)

    #mamy teraz train_pca i test_pca
    train_principal_Df = pd.DataFrame(data = train_pca
                , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

    test_principal_Df = pd.DataFrame(data = test_pca
                , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)


    X_train=train_principal_Df
    X_test=test_principal_Df 
    feature='male'

    input_dim = components_nr
    hidden_dim = 20
    output_dim = 1
    learning_rate = 0.075
    loss_fn = nn.BCELoss()
    num_epochs = 100
    seq_dim=1
    layer_dim=1

    if model_name=="svm":
    
        clf=train.svm_classification_model(X_train, y_train)
        accuracy, precision, recall, cm, fpr, tpr=test.svm_classification_model(X_test, y_test, clf)
    

    elif model_name=="forest":
  
        rf=train.random_forest_model(X_train, y_train, feature)
        accuracy, precision, recall, cm, fpr, tpr=test.random_forest_model(X_test, y_test, feature, rf)
        best_rf = rf.best_estimator_
        feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

        feature_importances.index = feature_importances.index.astype(int)
        print(feature_importances)
        #sort ascending by indexes
        feature_importances=feature_importances.sort_index()

        importance_df['component_names']=feature_importances.index
        importance_df['comp_imp']=feature_importances.values

 
    elif model_name=="rnn":
        learning_rate=1e-3
        model=train.recurrent_neural_network(X_train, y_train, seq_dim, input_dim, hidden_dim, layer_dim, output_dim, learning_rate, loss_fn, num_epochs)
        #print("expl",explainer)
        accuracy, precision, recall, FPR, cm, fpr, tpr=test.recurrent_neural_network_classification(X_test, y_test, seq_dim, input_dim, model)


    elif model_name=="nn":
        model=train.layer_neural_network(X_train, y_train, input_dim, hidden_dim, output_dim, learning_rate, loss_fn, num_epochs)
        accuracy, precision, recall, FPR, cm, fpr, tpr, df_fi=test.neural_network_classification(X_test, y_test, model)


    if model_name!="nn":
        if i==0:
            importance_df.to_csv(f'{results_directory}/importance_sex_{model_name}.csv', sep='\t')
        else:
            #concatenate
            importance_df_old=pd.read_csv(f'{results_directory}/importance_sex_{model_name}.csv', sep='\t', index_col=0)
            importance_df=pd.concat([importance_df_old, importance_df], axis=1)
            importance_df.to_csv(f'{results_directory}/importance_sex_{model_name}.csv', sep='\t', index=True)

    if model_name=="nn":
  
        if i==0:
            df_fi=pd.concat([df_fi.reset_index(drop=True), importance_df.reset_index(drop=True)], axis=1)
            df_fi.to_csv(f'{results_directory}/importance_sex_{model_name}.csv', sep='\t', index=False)
        else:
            #concatenate
            importance_df_old=pd.read_csv(f'{results_directory}/importance_sex_{model_name}.csv', sep='\t')
            importance_df=pd.concat([importance_df_old.reset_index(drop=True), df_fi.reset_index(drop=True), importance_df.reset_index(drop=True)], axis=1)
            importance_df.to_csv(f'{results_directory}/importance_sex_{model_name}.csv', sep='\t', index=False)
    

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
