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

for i in range(n_crosval):
    X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, label_names)
    X_train, X_test=prepare_dataset.standarize_data(X_train, X_test)

    #PCA
    components_nr=40
    pca_mri, train_pca, test_pca, component_loadings=dimensions_reduction.principal_component_analysis(X_train, X_test, components_nr)
    explained_variance_ratio=pca_mri.explained_variance_ratio_
    formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
    print('Explained variability per principal component: {}'.format(formatted_explained_variance))
    print(len(formatted_explained_variance))
    print(component_loadings.shape)
    print(train_pca.shape, test_pca.shape)
    #plots
    #plots.pca(pca_mri, train_pca, test_pca, X_train, y_train, X_test, y_test)
    #plots.scree_plot(pca_mri)

    #mamy teraz train_pca i test_pca
    train_principal_Df = pd.DataFrame(data = train_pca
                , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

    test_principal_Df = pd.DataFrame(data = test_pca
                , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)

    train_principal_Df['age']=X_train['age']
    test_principal_Df['age']=X_test['age']


    X_train=train_principal_Df
    X_test=test_principal_Df 
    feature='male'

    rf=train.random_forest_model(X_train, y_train, feature)
    accuracy, precision, recall, cm, fpr, tpr=test.random_forest_model(X_test, y_test, feature, rf)
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
