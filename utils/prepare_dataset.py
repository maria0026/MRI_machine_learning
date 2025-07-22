from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
from scipy.stats import randint, uniform
import pandas as pd
import yaml

class DatasetPreprocessor:
    def __init__(self):
        print("DatasetPreprocessor initialized")

    def detele_unnormal_columns(self, folder, folder_out, df_outliers):
        files=os.listdir(folder)
        for file in files:
            columns_to_drop=[]
            path=os.path.join(folder, file)
            print(path)
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
        #scaler = RobustScaler()

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

    def add_sex_column(self, df, principal_df):

        indices = principal_df.index.tolist()
        #add sex from df to principal_df 
        principal_df['male']=df.loc[indices]['male'].values.astype(int)

        return principal_df
    
    def load_model_config(self, model_name: str, config_file: str) -> tuple[dict, dict]:
        with open(config_file, "r") as f:
            full_config = yaml.safe_load(f)
        model_config = full_config.get(model_name, {})
        global_config = {k: v for k, v in full_config.items() if k != model_name}
        return global_config, model_config
    
    def convert_dist_params(self, param_dist):
        """Konwertuje wartości z YAML-a na obiekty scipy.stats, jeśli to konieczne"""
        dist = {}
        for key, val in param_dist.items():
            if isinstance(val, dict):
                if val.get("type") == "uniform":
                    low = val["low"]
                    high = val["high"]
                    dist[key] = uniform(loc=low, scale=high - low)
                elif val.get("type") == "randint":
                    low = val["low"]
                    high = val["high"]
                    dist[key] = randint(low, high)
            else:
                dist[key] = val
        return dist
