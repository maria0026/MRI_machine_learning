import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

def plot_some_data(df):
    #test normalnosci- bez kolumn typu płec 
    columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female']
    #anomalies_detection.test_normality(filename, columns_to_drop)
    df=df.drop(columns=columns_to_drop)
    #plot histograms for first 4 columns
    plt.figure(figsize=(10, 10))
    for i, column in enumerate(df.columns[:9]):
        plt.subplot(3, 3, i+1)
        plt.hist(df[column])
        plt.title(column)
    plt.subplots_adjust(hspace=0.5) 
    plt.show()

    plt.figure(figsize=(10, 10))
    for i, column in enumerate(df.columns[:9]):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x=df[column])
        plt.title(column)
    plt.subplots_adjust(hspace=1) 
    plt.show()

def pca(pca_mri, train_pca, test_pca, X_train, y_train, X_test, y_test):
    

    train_principal_Df = pd.DataFrame(data = train_pca
             , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)
    
    test_principal_Df = pd.DataFrame(data = test_pca
             , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)
    
    #stosunek wyjasnionej wariancji
    explained_variance_ratio=pca_mri.explained_variance_ratio_

    y_train['age_label']=y_train['age'].apply(lambda x: 1 if x>30 else 0)
    targets = [1,0]
    colors = ['r', 'g']

    plt.figure(figsize=(10,20))
    plt.subplot(1, 2, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1',fontsize=20)
    plt.ylabel('Principal Component 2',fontsize=20)
    plt.title("Train dataset",fontsize=20)
    for target, color in zip(targets,colors):
        indicesToKeep = y_train['male'] == target
        print(indicesToKeep)
        plt.scatter(train_principal_Df.loc[indicesToKeep, '1'], 
                    train_principal_Df.loc[indicesToKeep, '2'], 
                    c=color, s=50)
    plt.legend(targets,prop={'size': 15})


    plt.subplot(1, 2, 2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1',fontsize=20)
    plt.ylabel('Principal Component 2',fontsize=20)
    plt.title("Test dataset",fontsize=20)
   
    for target, color in zip(targets,colors):
        indicesToKeep = y_test['male'] == target
        plt.scatter(test_principal_Df.loc[indicesToKeep, '1'], 
                    test_principal_Df.loc[indicesToKeep, '2'], 
                    c=color, s=50)
    plt.legend(targets,prop={'size': 15})
    plt.show()

def scree_plot(pca_mri):

    PC_values = np.arange(pca_mri.n_components_) + 1
    plt.figure(figsize=(10,20))
    plt.subplot(1, 2, 1)
    plt.plot(PC_values, pca_mri.explained_variance_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')

    plt.subplot(1, 2, 2)
    plt.plot(PC_values, pca_mri.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component Ratio')
    plt.ylabel('Variance Explained')
    plt.show()

def random_forest(X_train, rf):


    # Export the first three decision trees from the forest
    for i in range(3):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(tree,
                                feature_names=X_train.columns,  
                                filled=True,  
                                max_depth=2, 
                                impurity=False, 
                                proportion=True)
        graph = graphviz.Source(dot_data)
        graph.view()

def confusion_matrix(cm):
    ConfusionMatrixDisplay(confusion_matrix=cm).plot();
    plt.show()

def random_forest_features(X_train, rf):
    # Create a series containing feature importances from the model and feature names from the training data
    best_rf = rf.best_estimator_
    feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Plot a simple bar chart
    feature_importances.plot.bar();
    plt.show()

def correlation_matrix(correlations):

  
    plt.figure(figsize=(15,10))
    sns.heatmap(round(correlations,2), cmap='RdBu',  
            vmin=-1, vmax=1);
    plt.show()
    
    