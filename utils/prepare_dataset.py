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
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

            return X_train, X_val, X_test, y_train, y_val, y_test 
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_val, y_val = None, None

            return X_train, X_val, X_test, y_train, y_val, y_test


    def standardize_data(self, X_train, X_val, X_test, column_to_copy=[]):
        scaler = StandardScaler()

        X_train_to_scale = X_train.drop(columns=column_to_copy)
        X_test_to_scale = X_test.drop(columns=column_to_copy)

        X_train_scaled = scaler.fit_transform(X_train_to_scale)
        X_test_scaled = scaler.transform(X_test_to_scale)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_to_scale.columns, index=X_train_to_scale.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_to_scale.columns, index=X_test_to_scale.index)

        X_train = pd.concat([X_train_scaled_df, X_train[column_to_copy]], axis=1)
        X_test = pd.concat([X_test_scaled_df, X_test[column_to_copy]], axis=1)

        if X_val is not None:
            X_val_to_scale = X_val.drop(columns=column_to_copy)
            X_val_scaled = scaler.transform(X_val_to_scale)
            X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val_to_scale.columns, index=X_val_to_scale.index)
            X_val = pd.concat([X_val_scaled_df, X_val[column_to_copy]], axis=1)
        else:
            X_val = None

        return X_train, X_val, X_test, scaler


    def divide_by_total_volume(self, df):

        for column in df.columns:
            if 'volume' in column or 'Volume' in column:
                df[column]=df[column]/df['Estimated_Total_Intracranial_Volume']

        return df

    def filter_by_sex(self, X_data, y_data, principal_df, sex_value):

        indices = X_data[X_data['male'] == sex_value].index.tolist()
        X_filtered = principal_df.loc[indices]
        y_filtered = y_data.loc[indices]
        return X_filtered, y_filtered
    
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
