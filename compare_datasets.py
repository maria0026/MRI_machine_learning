import pandas as pd
from utils import plots

filename='data_4/normal/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)
plots.plot_some_data(df)

filename='data_4/test_data_normal/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)
plots.plot_some_data(df)
