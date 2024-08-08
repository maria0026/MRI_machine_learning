import prepare_csv
import anomalies_detection
import pandas as pd
import os
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


'''
path = 'data_4/original/Subjects.csv'
prepare_csv.replace_comma_with_dot(path)

folder='data_4/original'
old_deliminer = ','
old_deliminer_2 = ';'
deliminer = '\t'

prepare_csv.convert_deliminer(folder, deliminer, old_deliminer, old_deliminer_2)
prepare_csv.convert_line_endings(folder)
columns=['Brain_Segmentation_Volume','Brain_Segmentation_Volume_Without_Ventricles','Brain_Segmentation_Volume_Without_Ventricles_from_Surf','Total_cortical_gray_matter_volume','Supratentorial_volume','Supratentorial_volume.1', 'Estimated_Total_Intracranial_Volume','Brain Segmentation Volume',	'Brain Segmentation Volume Without Ventricles']
#z wszystkich oprócz 1
filenames=['LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv','ASEG.csv', 'BRAIN.csv', 'WM.csv']
prepare_csv.delete_specified_columns(folder, filenames, columns)

columns=['White_Surface_Total_Area', 'Mean_Thickness']
filenames=['LHA2009.csv', 'LHAPARC.csv', 'RHA2009.csv', 'RHAPARC.csv']
prepare_csv.delete_specified_columns(folder, filenames, columns)

columns=['Volume of ventricles and choroid plexus']
filenames=['ASEG.csv']
prepare_csv.delete_specified_columns(folder, filenames, columns)

columns=['White_Surface_Total_Area', 'Mean_Thickness']
filenames=['LHDKT.csv', 'RHDKT.csv']
prepare_csv.add_hemisphere_name(folder, filenames, columns)

#uwuwanie duplikatów subjectów, brakujących danych i danych z norm_confirmed=0 na podstwaie Subjects.csv
filename='Subjects.csv'
indexes=prepare_csv.get_indexes_for_cleaning_dataset(folder, filename, data_files=False)


# Lista plików CSV
filenames = ['WM.csv', 'ASEG.csv', 'BRAIN.csv', 'LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv', 'RHDKT.csv']

for filename in filenames:
    indexes_df=prepare_csv.get_indexes_for_cleaning_dataset(folder,filename, data_files=True)
    if len(indexes_df)>0:
        indexes.append(indexes_df)
    else:
        print(f'No missing data in {filename}')

#delete duplicates from indexes

indexes = list(set(indexes))
print(indexes)
#czyszczenie danych


folder_out='data_4/cleaned_data'
prepare_csv.clean_datasets(indexes, folder, folder_out)

folder_name=folder_out
prepare_csv.convert_line_endings(folder_name)
prepare_csv.concatenate_datasets(folder_name)
'''

filename='data_4/cleaned_data/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)
#plots.plot_some_data(df)

columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age']
#anomalies_detection.test_normality(filename, columns_to_drop)

filename='outliers_values.csv'
df_outliers=pd.read_csv(filename, sep='\t',index_col=0)
row_sum = df_outliers.loc['wynik testu'].sum()
print("ilosc nienormalnych",row_sum)

#usunięcie nienormalnych danych
folder='data_4/cleaned_data'
folder_out='data_4/normal'
#prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)

filename='data_4/normal/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
#plots.plot_some_data(df)

#redukcja wymiarowości

folder='data_4/normal'
folder_out='data_4/correlation_matrices'
#w konkretnych csv
#dimensions_reduction.calculate_correlation_matrices(folder, folder_out)

#połączenie we wszystkie możliwe pary
folder_out='data_4/pairs'
#dimensions_reduction.pair_files(folder, folder_out)
folder=folder_out
folder_out='data_4/correlation_matrices_pairs'
#dimensions_reduction.calculate_correlation_matrices(folder, folder_out)


#podział na zbiór treningowy i testowy
df=pd.read_csv('data_4/normal/all_concatenated.csv', sep='\t')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])
label_names=['male', 'age']
X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, label_names)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train, X_test=prepare_dataset.standarize_data(X_train, X_test)

#PCA
components_nr=40
pca_mri, train_pca, test_pca=dimensions_reduction.principal_component_analysis(X_train, X_test, components_nr)
explained_variance_ratio=pca_mri.explained_variance_ratio_
formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
print('Explained variability per principal component: {}'.format(formatted_explained_variance))
#plots
#plots.pca(pca_mri, train_pca, test_pca, X_train, y_train, X_test, y_test)
#plots.scree_plot(pca_mri)

#mamy teraz train_pca i test_pca
train_principal_Df = pd.DataFrame(data = train_pca
            , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

test_principal_Df = pd.DataFrame(data = test_pca
            , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)
    
#wytrenowanie po pca
feature='male'
X_train=train_principal_Df
X_test=test_principal_Df

rf=train.random_forest_model(X_train, y_train, feature)
accuracy, precision, recall, cm=test.random_forest_model(X_test, y_test, feature, rf)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
plots.confusion_matrix(cm)
plots.random_forest_features(X_train, rf)

#policzenie korelacji dla całego zbioru
'''
path='data_4/normal/all_concatenated.csv'
folder_out='data_4/correlation_matrices'
df=pd.read_csv(path, sep='\t')
df=df.dropna(axis=1, how='all')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female', 'male', 'age'])
corr_matrix=df.corr()
corr_matrix.to_csv(f'{folder_out}/all_concatenated_correlation_matrix.csv', sep='\t', index=True)
'''

#wyswietlenie korelacji
correlations=pd.read_csv('data_4/correlation_matrices/all_concatenated_correlation_matrix.csv', sep='\t', index_col=0)
correlations=correlations.iloc[:50,:50]
plots.correlation_matrix(correlations)
clustered_correlations=dimensions_reduction.cluster_correlations(correlations)
plots.correlation_matrix(clustered_correlations)

