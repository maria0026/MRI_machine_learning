import pandas as pd
import numpy as np
from scipy import stats

class AnomaliesDetector:
    def __init__(self):
        print("AnomaliesDetector initialized")
        
    def test_normality(self, filename, columns_to_drop):
        df=pd.read_csv(filename, sep='\t')
        df = df[(df['age'] <= 100) & (df['age'] >= 0.5)]
        df = df[~((df['age'] < 1) & (df['hight'] > 100))]
        df=df.drop(columns=columns_to_drop)
        df = df.fillna(0)

        #df for outliers marking
        df_outliers=pd.DataFrame(columns=df.columns, index=df.index)
        normality_scores = {}

        for column in df.columns:
            data=df[column]
            #norality test before outlier detection
            std_all = np.std(data)

            # Test normalności przed usunięciem outlierów
            if std_all == 0 or np.isnan(std_all):
                p = np.nan
            else:
                try:
                    _, p = stats.kstest(data, 'norm', args=(np.mean(data), std_all))
                    #_, p = stats.normaltest(data)
                except Exception:
                    p = np.nan


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
            data_no_outliers=data[df_outliers[column]==0]
            std_clean = np.std(data_no_outliers)

            if len(data_no_outliers) < 5 or std_clean == 0 or np.isnan(std_clean):
                p_after = np.nan
                test_result = np.nan
            else:
                try:
                    _, p_after = stats.kstest(data_no_outliers, 'norm',
                                            args=(np.mean(data_no_outliers), np.std(data_no_outliers)))
                    #_, p_after = stats.normaltest(data_no_outliers)
                    test_result = int(p_after < 0.05)
                except Exception:
                    p_after = np.nan
                    test_result = np.nan
    
            #sum number of outliers in column
            outliers_number=df_outliers[column].sum()
            normality_scores[column]=[median, mad, outliers_number, p, p_after, test_result]

        df_normality_scores = pd.DataFrame(normality_scores)
        df_normality_scores.insert(0, 'name', ['median', 'mad', 'outliers_number', 'p', 'p_after', 'wynik testu'])
        df_normality_scores.set_index('name', inplace=True)

        return df_normality_scores, df_outliers

    
 

