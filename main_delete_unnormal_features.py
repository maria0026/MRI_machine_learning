from utils import anomalies_detection
from utils import prepare_dataset
import pandas as pd


norm_confimed=3

if norm_confimed==1:
    type='positive'
elif norm_confimed==0:
    type='negative'
else:
    type='all'

filename=f'data/{type}_norm_confirmed/all_concatenated.csv'


df=pd.read_csv(filename, sep='\t')
print(df)
#plots.plot_some_data(df)

columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume']

#for sex identyfication
df_mad, df_outliers= anomalies_detection.test_normality(filename, columns_to_drop)
df_mad.to_csv(f'results/{type}_outliers_values.csv', sep='\t', index=False)
df_outliers.to_csv(f'results/{type}_outliers.csv', sep='\t', index=True)

filename=f'results/{type}_outliers_values.csv'
df_outliers=pd.read_csv(filename, sep='\t',index_col=0)
row_sum = df_outliers.loc['wynik testu'].sum()
print("ilosc nienormalnych",row_sum)

#usunięcie nienormalnych danych
folder=f'data/{type}_norm_confirmed'
folder_out=f'data/{type}_norm_confirmed_normal'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)

folder=f'data/negative_norm_confirmed'
folder_out=f'data/negative_norm_confirmed_normal'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)

