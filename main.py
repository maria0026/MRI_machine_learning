import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import numpy as np
from utils import dimensions_reduction
from utils import prepare_dataset
from utils import plots
from sklearn.metrics import roc_curve, auc
from utils import volume_analysis


filename='data/positive_norm_confirmed/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
#print(df)
#plots.plot_some_data(df)

#macierze korelacji z Total Intracranial Volume
path='data/positive_norm_confirmed_normal/all_concatenated.csv'
folder_out='data/correlation_matrices'
#volume_analysis.correlation_matrix_total_volume(path, folder_out)


#wyswietlenie korelacji
correlations=pd.read_csv('data/correlation_matrices/volume_area_correlation_matrix.csv', sep='\t', index_col=0)
#wykres osobno dla mężczyzn i kobiet
df=prepare_dataset.transform_correlations_total_volume(correlations)
#plots.correlations_total_volume(df)


#macierze korelacji

'''
folder='data/normal'
#w konkretnych csv
dimensions_reduction.calculate_correlation_matrices(folder, folder_out)

#połączenie we wszystkie możliwe pary
folder_out='data/pairs'
dimensions_reduction.pair_files(folder, folder_out)
folder=folder_out
folder_out='data/correlation_matrices_pairs'
dimensions_reduction.calculate_correlation_matrices(folder, folder_out)
'''

'''
path='data/normal/all_concatenated.csv'
folder_out='data/correlation_matrices'
df=pd.read_csv(path, sep='\t')
df=df.dropna(axis=1, how='all')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female', 'male', 'age'])
#prepare_dataset.divide_by_total_volume(df)
corr_matrix=df.corr()
corr_matrix.to_csv(f'{folder_out}/all_concatenated_volume_devided_correlation_matrix.csv', sep='\t', index=True)

correlations=pd.read_csv('data/correlation_matrices/all_concatenated_volume_devided_correlation_matrix.csv', sep='\t', index_col=0)
clustered_correlations, selected_features=dimensions_reduction.cluster_correlations(correlations)
print(selected_features)

'''
#wyswietlenie korelacji
correlations=pd.read_csv('data/correlation_matrices/all_concatenated_correlation_matrix.csv', sep='\t', index_col=0)
correlations=correlations.iloc[:50,:50]
#plots.correlation_matrix(correlations)
#clustered_correlations, selected_featues=dimensions_reduction.cluster_correlations(correlations)
#plots.correlation_matrix(clustered_correlations)


importance_df=pd.read_csv('results/importance_sex_nn.csv', sep='\t')
importance_df=pd.read_csv('results/importance_sex_forest.csv', sep='\t')
#plots.component_importance(importance_df, model='Tree')

#age prediction results
results_nn=pd.read_csv('results/regression_results_nn.csv', sep='\t')
plots.age_prediction_function(results_nn, model="Neural Network")

results_svm=pd.read_csv('results/regression_results_svm.csv', sep='\t')
plots.age_prediction_function(results_svm, model="SVM")

results_tree=pd.read_csv('results/regression_results_forest.csv', sep='\t')
plots.age_prediction_function(results_tree, model="Forest")

results_rnn=pd.read_csv('results/regression_results_rnn.csv', sep='\t')
plots.age_prediction_function(results_rnn, model="RNN")