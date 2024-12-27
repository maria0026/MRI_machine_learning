import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import auc

def plot_some_data(df):
    
    columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female']
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

def pca(train_pca, test_pca, X_train, y_train, X_test, y_test):
    

    train_principal_Df = pd.DataFrame(data = train_pca
             , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)
    
    test_principal_Df = pd.DataFrame(data = test_pca
             , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)
    

    #y_train['age_label']=y_train['age'].apply(lambda x: 1 if x>30 else 0)
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

    
def scree_plot(pca_mri, type='positive', dev=False):
    PC_values = np.arange(pca_mri.n_components_) + 1

    plt.style.use('default')
    plt.plot(PC_values, pca_mri.explained_variance_ratio_, 'o-',  linewidth=2, color='m')
    plt.title('Scree Plot', fontsize=16)
    plt.xlabel('Principal Component', fontsize=16)
    plt.ylabel('Variance Explained Ratio', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if dev:
        plt.savefig(f'plots/scree_plot_{type}_dev.png')
    else:
        plt.savefig(f'plots/scree_plot_{type}.png')
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
    best_rf = rf.best_estimator_
    feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_importances.plot.bar();
    plt.show()


def correlation_matrix(correlations):
    plt.figure(figsize=(20, 20))
    sns.heatmap(round(correlations,2), cmap='RdBu',  
            vmin=-1, vmax=1);
    plt.show()

def roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)
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
    
def t_sne(train_pca, X_train, y_train):
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

#compare distributions in train and test set
def distribution(train_pca, test_pca): 

    train_pca=pd.DataFrame(train_pca)
    test_pca=pd.DataFrame(test_pca)
    print(train_pca)
    train_pca.columns = [str(i) for i in range(1,train_pca.shape[1]+1)]
    test_pca.columns = [str(i) for i in range(1,test_pca.shape[1]+1)]
    plt.figure(figsize=(10, 10))
    for i, column in enumerate(test_pca.columns[:9]):
        plt.subplot(3, 3, i+1)
        min_value = min(train_pca[column].min(), test_pca[column].min())
        max_value = max(train_pca[column].max(), test_pca[column].max())
        bins = np.linspace(min_value, max_value, 30)
        
        plt.hist(test_pca[column], color='red', alpha=0.5, label='Test')
        plt.hist(train_pca[column], color='green', alpha=0.5, label='Train')
        plt.title(column)
        plt.legend(loc='upper right')
    plt.subplots_adjust(hspace=0.5) 
    plt.show()


def age_prediction_function(df, model, data_type, valid=False, test=''):
    # Tworzenie figury z odpowiednim rozmiarem
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    #fig.suptitle(f'Chronological vs Predicted Age by {model}', fontsize=16)
    list_actual = [column for column in df.columns if 'Actual' in column]
    list_predicted = [column for column in df.columns if 'Predicted' in column]

    all_actual = np.array([])
    all_predicted = np.array([])

    for i in range(len(list_actual)):
        ax = axs[i // 3, i % 3]  # Pobieranie odpowiedniego podwykresu (2x3 layout)
        ax.plot(df[list_actual[i]], df[list_predicted[i]], 'o', alpha=0.3)
        ax.set_xlabel('Chronological Age (years)', fontsize=12)
        ax.set_ylabel('Predicted Age (years)', fontsize=12)
        ax.set_title(f'Training {i + 1}', fontsize=14)
        ax.tick_params(axis='both', labelsize=10)


        all_actual = np.append(all_actual, df[list_actual[i]].values)
        all_predicted = np.append(all_predicted, df[list_predicted[i]].values)

    all_actual=list(all_actual)
    all_predicted=list(all_predicted)
    # Tworzenie figury i osi
    ax=axs[1,2]  
    ax.plot(all_actual, all_predicted, 'o', alpha=0.2)
    unique_actual = np.unique(all_actual)
    mean_predicted = [np.mean([all_predicted[j] for j in range(len(all_actual)) if all_actual[j] == ua]) for ua in unique_actual]

    # Dopasowanie wielomianu do unikalnych danych
    z = np.polyfit(unique_actual, mean_predicted, 2)
    p = np.poly1d(z)

    ax.plot(unique_actual, p(unique_actual), "r--", label='y=%.2fx² + %.2fx + %.2f' % (z[0], z[1], z[2]))
    #add line y=x
    ax.plot(unique_actual, unique_actual, "b--", color='m', label='y=x')
    ax.legend()
    ax.set_xlabel('Chronological Age (years)')
    ax.set_ylabel('Predicted Age (years)')
    ax.set_title(f'All trains by {model}')


    fig.tight_layout()  
    if valid:
        plt.savefig(f'plots/{data_type}_{model}_valid_{test}.png')
    else:
        plt.savefig(f'plots/{data_type}_{model}_plot_{test}.png')
    plt.show()

def age_prediction_gap(df, model, data_type, valid=False, test=''):
    # Tworzenie figury z odpowiednim rozmiarem
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f'Gap in prediction by {model}', fontsize=16)
    list_actual = [column for column in df.columns if 'Actual' in column]
    list_predicted = [column for column in df.columns if 'Predicted' in column]

    all_actual = np.array([])
    all_predicted_dif = np.array([])

    for i in range(len(list_actual)):
        ax = axs[i // 3, i % 3]  # Pobieranie odpowiedniego podwykresu (2x3 layout)
        ax.plot(df[list_actual[i]], df[list_predicted[i]]-df[list_actual[i]],  'o', alpha=0.3, color='orange')
        ax.set_xlabel('Chronological Age (years)', fontsize=12)
        ax.set_ylabel('Age gap (years)', fontsize=12)
        ax.set_title(f'Training {i + 1}', fontsize=14)
        ax.tick_params(axis='both', labelsize=10)


        all_actual = np.append(all_actual, df[list_actual[i]].values)
        all_predicted_dif = np.append(all_predicted_dif, df[list_predicted[i]].values-df[list_actual[i]].values)

    all_actual=list(all_actual)
    all_predicted_dif=list(all_predicted_dif)
    # Tworzenie figury i osi
    ax=axs[1,2]  
    ax.plot(all_actual, all_predicted_dif, 'o', alpha=0.2, color='orange')

    unique_actual = np.unique(all_actual)
    mean_predicted_dif = [np.mean([all_predicted_dif[j] for j in range(len(all_actual)) if all_actual[j] == ua]) for ua in unique_actual]

    # Dopasowanie wielomianu do unikalnych danych
    z = np.polyfit(unique_actual, mean_predicted_dif, 2)
    p = np.poly1d(z)

    # Tworzenie wykresu dla wielomianu kwadratowego
    ax.plot(unique_actual, p(unique_actual), "r--", label='y=%.2fx² + %.2fx + %.2f' % (z[0], z[1], z[2]))

    ax.legend()
    ax.set_xlabel('Chronological Age (years)')
    ax.set_ylabel('Age gap (years)')
    ax.set_title(f'Chronological vs Predicted Age by {model}')


    fig.tight_layout()  
    if valid:
        plt.savefig(f'plots/{data_type}_{model}_gap_valid_{test}.png')
    else:
        plt.savefig(f'plots/{data_type}_{model}_gap_plot_{test}.png')
    plt.show()
    plt.show()

def age_prediction_gap_2_atlases(df1, df2, atlas1, atlas2, model, data_type, valid=False, test=''):

    list_actual_1= [column for column in df1.columns if 'Actual' in column][0]
    list_predicted = [column for column in df1.columns if 'Predicted' in column][0]
    list_actual_2 = [column for column in df2.columns if 'Actual' in column][0]
    list_predicted_2 = [column for column in df2.columns if 'Predicted' in column][0]
    gap_1 = df1[list_predicted]-df1[list_actual_1]
    gap_2 = df2[list_predicted_2]-df2[list_actual_2]
    plt.figure(figsize=(10, 5))
    #plt.plot(df1[list_actual_1], gap_1, 'o', alpha=0.5, label=f'{atlas1}')
    #plt.plot(df2[list_actual_2], gap_2, 'o', alpha=0.5, label=f'{atlas2}')
    plt.scatter(gap_1, gap_2, alpha=0.5)
    plt.xlabel(f'Age gap (years) {atlas2}', fontsize=12)
    plt.ylabel(f'Age gap (years) {atlas1}', fontsize=12)
    plt.legend()
    plt.show()


