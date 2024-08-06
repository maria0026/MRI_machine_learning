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

'''
path = 'data_4/original/Subjects.csv'
prepare_csv.replace_comma_with_dot(path)

folder='data_4/original'

#prepare_csv.merge_by_atlas(folder, 'A2009')
#prepare_csv.merge_by_atlas(folder, 'APARC')
#prepare_csv.merge_by_atlas(folder, 'DKT')

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
#print(indexes)


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

#test normalnosci- bez kolumn typu płec 
#columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female']
#anomalies_detection.test_normality(filename, columns_to_drop)

'''
stat, p = shapiro(df['APARC-ctx-lh-bankssts_ThickAvg'])
print(stat,p)
plt.hist(df['age'])
plt.show()

sns.boxplot(x=df['age'])
plt.show()
z = np.abs(stats.zscore(df['age']))
print(z)
threshold_z = 2
 
outlier_indices = np.where(z > threshold_z)[0]
no_outliers = df.drop(outlier_indices)
print("Original DataFrame Shape:", df.shape)
print("DataFrame Shape after Removing Outliers:", no_outliers.shape)
'''

#redukcja wymiarowości
folder='data_4/cleaned_data'
folder_out='data_4/correlation_matrices'
#w konkretnych csv
#dimensions_reduction.calculate_correlation_matrices(folder, folder_out)

#połączenie we wszystkie możliwe pary
folder_out='data_4/pairs'
#dimensions_reduction.pair_files(folder, folder_out)
folder=folder_out
folder_out='data_4/correlation_matrices_pairs'
#dimensions_reduction.calculate_correlation_matrices(folder, folder_out)


#wyswietlenie korelacji
df=pd.read_csv('data_4/correlation_matrices/BRAIN_correlation_matrix.csv', sep='\t', index_col=0)
sns.heatmap(df, annot=True)
plt.show()


#podział na zbiór treningowy i testowy

df=pd.read_csv('data_4/cleaned_data/all_concatenated.csv', sep='\t')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])
label_names=['male', 'age']
X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, label_names)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
prepare_dataset.standarize_data(X_train, X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#PCA
dimensions_reduction.principal_component_analysis(X_train, X_test)

