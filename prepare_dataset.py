from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

def detele_unnormal_columns(folder, folder_out, df_outliers):
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
        df.to_csv(f'{folder_out}/{file}', sep='\t', index=False)
    return df

    
def split_dataset(df, label_names):
    X=df.drop(columns=label_names)
    y=df[label_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def standarize_data(X_train, X_test):
    
    sc = StandardScaler()
    X_train_standarized = sc.fit_transform(X_train)
    X_test_standarized = sc.transform(X_test)
    X_train = pd.DataFrame(X_train_standarized, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_standarized, columns=X_test.columns, index=X_test.index)
    return X_train, X_test