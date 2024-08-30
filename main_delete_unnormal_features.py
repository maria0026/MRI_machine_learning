from utils import anomalies_detection
from utils import prepare_dataset
import pandas as pd


filename='data/test_data_all/all_concatenated.csv'

filename='data/cleaned_data/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)
#plots.plot_some_data(df)

columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume']

#for sex identyfication
anomalies_detection.test_normality(filename, columns_to_drop)

filename='results/outliers_values.csv'
df_outliers=pd.read_csv(filename, sep='\t',index_col=0)
row_sum = df_outliers.loc['wynik testu'].sum()
print("ilosc nienormalnych",row_sum)

#usunięcie nienormalnych danych
folder='data/cleaned_data'
folder_out='data/normal'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)

folder='data/test_data'
folder_out='data/test_data_normal'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)

'''
#unnormal from test data
filename='data/test_data_normal/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)
#plots.plot_some_data(df)

columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume']
anomalies_detection.test_normality(filename, columns_to_drop)

filename='results/outliers_values.csv'
df_outliers=pd.read_csv(filename, sep='\t',index_col=0)
row_sum = df_outliers.loc['wynik testu'].sum()
print("ilosc nienormalnych",row_sum)

#usunięcie nienormalnych danych
folder='data/normal'
folder_out='data/normal'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)

folder='data/test_data_normal'
folder_out='data/test_data_normal'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)
'''

'''
folder='data/test_data_all'
folder_out='data/all'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)
'''