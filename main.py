import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import numpy as np
import dimensions_reduction
import prepare_dataset
import plots
from sklearn.metrics import roc_curve, auc
import volume_analysis


filename='data_4/cleaned_data/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)
#plots.plot_some_data(df)


#macierze korelacji z Total Intracranial Volume
path='data_4/cleaned_data/all_concatenated.csv'
folder_out='data_4/correlation_matrices'
volume_analysis.correlation_matrix_total_volume(path, folder_out)

#wyswietlenie korelacji

correlations=pd.read_csv('data_4/correlation_matrices/volume_area_correlation_matrix.csv', sep='\t', index_col=0)
#wykres osobno dla mężczyzn i kobiet
df=prepare_dataset.transform_correlations_total_volume(correlations)
#plots.correlations_total_volume(df)


#macierze korelacji

'''
folder='data_4/normal'
#w konkretnych csv
dimensions_reduction.calculate_correlation_matrices(folder, folder_out)

#połączenie we wszystkie możliwe pary
folder_out='data_4/pairs'
dimensions_reduction.pair_files(folder, folder_out)
folder=folder_out
folder_out='data_4/correlation_matrices_pairs'
dimensions_reduction.calculate_correlation_matrices(folder, folder_out)
'''

path='data_4/normal/all_concatenated.csv'
folder_out='data_4/correlation_matrices'
df=pd.read_csv(path, sep='\t')
df=df.dropna(axis=1, how='all')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female', 'male', 'age'])
#prepare_dataset.divide_by_total_volume(df)
corr_matrix=df.corr()
corr_matrix.to_csv(f'{folder_out}/all_concatenated_volume_devided_correlation_matrix.csv', sep='\t', index=True)


correlations=pd.read_csv('data_4/correlation_matrices/all_concatenated_volume_devided_correlation_matrix.csv', sep='\t', index_col=0)
correlations=pd.read_csv('data_4/correlation_matrices/all_concatenated_correlation_matrix.csv', sep='\t', index_col=0)
clustered_correlations, selected_features=dimensions_reduction.cluster_correlations(correlations)
print(selected_features)

#wyswietlenie korelacji

correlations=pd.read_csv('data_4/correlation_matrices/all_concatenated_correlation_matrix.csv', sep='\t', index_col=0)
correlations=correlations.iloc[:50,:50]
plots.correlation_matrix(correlations)
clustered_correlations, selected_featues=dimensions_reduction.cluster_correlations(correlations)
plots.correlation_matrix(clustered_correlations)
