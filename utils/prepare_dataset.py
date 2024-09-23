from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

class DatasetPreprocessor:
    def __init__(self):
        print("DatasetPreprocessor initialized")

    def detele_unnormal_columns(self, folder, folder_out, df_outliers):
        files=os.listdir(folder)
        for file in files:
            columns_to_drop=[]
            path=os.path.join(folder, file)
            df=pd.read_csv(path, sep='\t')
        
            for column in df.columns:
                if column in df_outliers.columns:
                    if df_outliers.loc['wynik testu', column]==1:
                        columns_to_drop.append(column)
            df=df.drop(columns=columns_to_drop)
            df=df.dropna(axis=1, how='all')
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
            df.to_csv(f'{folder_out}/{file}', sep='\t', index=False)

        
    def split_dataset(self, df, label_names, test_size=0.2, valid=False):
        X=df.drop(columns=label_names)
        y=df[label_names]
        
        if valid:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.5)

            return X_train, X_val, X_test, y_train, y_val, y_test 
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_val, y_val = None, None

            return X_train, X_val, X_test, y_train, y_val, y_test

    def standarize_data(self, X_train, X_test, valid=False, X_val=None):
        
        sc = StandardScaler()
        X_train_standarized = sc.fit_transform(X_train)
        X_test_standarized = sc.transform(X_test)
        X_train = pd.DataFrame(X_train_standarized, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_standarized, columns=X_test.columns, index=X_test.index)
        if valid:
            X_val_standarized = sc.transform(X_val)
            X_val = pd.DataFrame(X_val_standarized, columns=X_val.columns, index=X_val.index)
            return X_train, X_val, X_test

        return X_train, None, X_test

    def divide_by_total_volume(self, df):

        for column in df.columns:
            if 'volume' in column or 'Volume' in column:
                df[column]=df[column]/df['Estimated_Total_Intracranial_Volume']

        return df

    #nieuzywane
    def transform_correlations_total_volume(self, correlations):

        # Inicjalizacja słownika dla przekształconych danych
        data = {'Metric': [], 'Male': [], 'Female': []}

        # Iteracja po wierszach danych
        for metric in correlations.index:
            if 'female' in metric:
                # Usunięcie '_female' z nazwy metryki
                base_metric = metric.replace('_female', '')
                
                # Sprawdzenie czy istnieje odpowiadająca metryka dla mężczyzn
                male_metric = correlations.index[correlations.index.str.contains(base_metric) & ~correlations.index.str.contains('female')]
                if len(male_metric) > 0:
                    male_value = correlations.loc[male_metric[0]].values[0]
                    female_value = correlations.loc[metric].values[0]
                    
                    # Dodanie wartości do słownika
                    data['Metric'].append(base_metric)
                    data['Male'].append(male_value)
                    data['Female'].append(female_value)

        df = pd.DataFrame(data)
        return df
