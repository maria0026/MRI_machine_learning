from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_dataset(df, label_names):
    X=df.drop(columns=label_names)
    y=df[label_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def standarize_data(X_train, X_test):
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test