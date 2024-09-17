import pandas as pd
import numpy as np
from scipy import stats

def test_normality(filename, columns_to_drop):
    df=pd.read_csv(filename, sep='\t')
    df=df.drop(columns=columns_to_drop)

    #df for outliers marking
    df_outliers=pd.DataFrame(columns=df.columns, index=df.index)
    normality_scores = {}

    for column in df.columns:
        data=df[column]
        #norality test before outlier detection
        _, p= stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))

        #outliers detection
        mad=stats.median_abs_deviation(data)
        median = np.median(data)
        threshold_mad = 3
        
        upper_bound = median + threshold_mad * mad
        lower_bound = median - threshold_mad * mad

        #marking outliers in df_outliers
        df_outliers[column] = ((data > upper_bound) | 
                             (data < lower_bound)).astype(int)
        
        #choose only data without outliers
        data=data[df_outliers[column]==0]
        
        _, p_after= stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))

        test_result=0
        if p_after < 0.05:
            test_result=1
        else:
            test_result=0
        
        #sum number of outliers in column
        outliers_number=df_outliers[column].sum()
        normality_scores[column]=[median, mad, outliers_number, p, p_after, test_result]

    df_normality_scores = pd.DataFrame(normality_scores)
    df_normality_scores.insert(0, 'name', ['median', 'mad', 'outliers_number', 'p', 'p_after', 'wynik testu'])
    df_normality_scores.set_index('name', inplace=True)

    return df_normality_scores, df_outliers

    
 

