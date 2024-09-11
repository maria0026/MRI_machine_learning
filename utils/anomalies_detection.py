import pandas as pd
import numpy as np
from scipy.stats import shapiro 
from scipy import stats

def test_normality(filename, columns_to_drop):
    df=pd.read_csv(filename, sep='\t')
    df_test=df.drop(columns=columns_to_drop)
    wynik=0
    mad_values = {}
    df_outliers=pd.DataFrame(columns=df_test.columns, index=df_test.index)

    for column in df_test.columns:
        data=df_test[column]

        #wykrycie outlierów
        mad=stats.median_abs_deviation(data)
        median = np.median(data)
        threshold_mad = 3
        
        #test normalności
        #stat, p = shapiro(data)
        #stat, p=sm.stats.diagnostic.lilliefors(data)
        stat, p= stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))

        upper_bound = median + threshold_mad * mad
        lower_bound = median - threshold_mad * mad

        #Oznaczanie outlierów w DataFrame
        df_outliers[column] = ((data > upper_bound) | 
                             (data < lower_bound)).astype(int)
        
        #sum number of 1 in column
        outliers_number=df_outliers[column].sum()
        
        data=data[df_outliers[column]==0]

        #stat, p_after = shapiro(data)
        #stat, p_after=sm.stats.diagnostic.lilliefors(data)
        stat, p_after= stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        
        if p_after < 0.05:
            wynik=1
        else:
            wynik=0
        mad_values[column]=[median, mad, outliers_number, p, p_after, wynik]

    df_mad = pd.DataFrame(mad_values)

    df_mad.insert(0, 'name', ['median', 'mad', 'outliers_number', 'p', 'p_after', 'wynik testu'])
    return df_mad, df_outliers

    
 

