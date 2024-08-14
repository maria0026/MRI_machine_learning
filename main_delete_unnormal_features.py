import anomalies_detection
import prepare_dataset
import pandas as pd

filename='data_4/cleaned_data/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)
#plots.plot_some_data(df)

filename='data_4/cleaned_data/all_concatenated.csv'
columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume']

#for sex identyfication
anomalies_detection.test_normality(filename, columns_to_drop)

filename='outliers_values.csv'
df_outliers=pd.read_csv(filename, sep='\t',index_col=0)
row_sum = df_outliers.loc['wynik testu'].sum()
print("ilosc nienormalnych",row_sum)

#usunięcie nienormalnych danych
folder='data_4/cleaned_data'
folder_out='data_4/normal'
prepare_dataset.detele_unnormal_columns(folder, folder_out, df_outliers)