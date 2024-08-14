import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.io as pio

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
    

    y_train['age_label']=y_train['age'].apply(lambda x: 1 if x>30 else 0)
    targets = [1,0]
    colors = ['r', 'g']

    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1',fontsize=20)
    plt.ylabel('Principal Component 2',fontsize=20)
    plt.title("Train dataset PCA",fontsize=20)
    for target, color in zip(targets,colors):
        indicesToKeep = y_train['male'] == target
        print(indicesToKeep)
        plt.scatter(train_principal_Df.loc[indicesToKeep, '1'], 
                    train_principal_Df.loc[indicesToKeep, '2'], 
                    c=color, s=50)
    plt.legend(['male', 'female'],prop={'size': 15})


    plt.subplot(1, 2, 2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1',fontsize=20)
    plt.ylabel('Principal Component 2',fontsize=20)
    plt.title("Test dataset PCA",fontsize=20)
   
    for target, color in zip(targets,colors):
        indicesToKeep = y_test['male'] == target
        plt.scatter(test_principal_Df.loc[indicesToKeep, '1'], 
                    test_principal_Df.loc[indicesToKeep, '2'], 
                    c=color, s=50, cmap='viridis')
    plt.legend(['male', 'female'],prop={'size': 15})
    plt.show()

    
def scree_plot(pca_mri):

    PC_values = np.arange(pca_mri.n_components_) + 1
    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.plot(PC_values, pca_mri.explained_variance_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')

    plt.subplot(1, 2, 2)
    plt.plot(PC_values, pca_mri.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained Ratio')
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

  
    plt.figure(figsize=(20, 20))
    sns.heatmap(round(correlations,2), cmap='RdBu',  
            vmin=-1, vmax=1);
    plt.show()

def roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    
def t_sne(pca_mri, train_pca, X_train, y_train):
    

    train_principal_Df = pd.DataFrame(data = train_pca
             , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

    

    y_train['age_label']=y_train['age'].apply(lambda x: 1 if x>30 else 0)
    targets = [1,0]
    colors = ['r', 'g']

    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1',fontsize=20)
    plt.ylabel('Principal Component 2',fontsize=20)
    plt.title("Train dataset- t_SNE",fontsize=20)
    for target, color in zip(targets,colors):
        indicesToKeep = y_train['male'] == target
        print(indicesToKeep)
        plt.scatter(train_principal_Df.loc[indicesToKeep, '1'], 
                    train_principal_Df.loc[indicesToKeep, '2'], 
                    c=color, s=50)
    plt.legend(['male', 'female'],prop={'size': 15})
    plt.show()

def roc_curve_crossvalid(tprs, fprs, aucs, feature):

    mean_fpr=np.mean(fprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print("srednie pole", mean_auc)
    std_auc = np.std(aucs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label '{feature}')",
    )
    ax.legend(loc="lower right")
    plt.show()

def correlations_total_volume(df):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 10))
    df.set_index('Metric').plot(kind='bar', ax=ax)
    ax.set_ylabel('Correlation with Intracranial Volume')
    ax.set_title('Gender-based Correlation Differences')
    ax.legend(title='Gender')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
