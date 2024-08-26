from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import nn_data
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.base import BaseEstimator, ClassifierMixin

def random_forest_model(X_test, y_test, feature, rf):
    
    # Create a variable for the best model
    best_rf = rf.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rf.best_params_)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test[feature], y_pred)
    precision = precision_score(y_test[feature], y_pred)
    recall = recall_score(y_test[feature], y_pred)


    # Create the confusion matrix
    cm = confusion_matrix(y_test[feature], y_pred)

    #roc curve
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[feature], y_pred_proba) 
    

    return accuracy, precision, recall, cm, fpr, tpr

def random_forest_regression_model(X_test, y_test, feature, rf):

    # Create a variable for the best model
    best_rf = rf.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rf.best_params_)

    y_pred = rf.predict(X_test)
    #export y_test and y_pred to df and then to csv
    print(y_test[feature].shape, y_pred.shape)
    # Make sure y_test[feature] and y_pred are aligned
    results_df = pd.DataFrame({
        'Actual': y_test[feature].values,  # Convert to NumPy array if it's not already
        'Predicted': y_pred
    })
    results_df.to_csv('results.csv')
    
    mse=mean_squared_error(y_test[feature], y_pred)
    rmse = float(format(np.sqrt(mean_squared_error(y_test[feature], y_pred)), '.3f'))

    #mean average error
    mae = mean_absolute_error(y_test[feature], y_pred)

    return mse, rmse, mae

def svm_classification_model(X_test, y_test, clf):

    # Create a variable for the best model
    best_rf = clf.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  clf.best_params_)


    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    #roc curve
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
    

    return accuracy, precision, recall, cm, fpr, tpr

def svm_regression_model(X_test, y_test, clf):

    # Create a variable for the best model
    best_rf = clf.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  clf.best_params_)


    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Make sure y_test[feature] and y_pred are aligned
    results_df = pd.DataFrame({
        'Actual': y_test.values.ravel(),  # Convert to NumPy array if it's not already
        'Predicted': y_pred.ravel()
    })
    results_df.to_csv('results_svm.csv')
    
    mse=mean_squared_error(y_test, y_pred)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))

    #mean average error
    mae = mean_absolute_error(y_test, y_pred)

    return mse, rmse, mae

class SklearnPyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        # Since the model is already trained, no fitting is needed here.
        return self
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return np.where(outputs.numpy().flatten() < 0.5, 0, 1)
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return np.vstack([1 - outputs.numpy().flatten(), outputs.numpy().flatten()]).T

def neural_network_classification(X_test, y_test, model):

    batch_size = 64
    test_data =nn_data.Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    total=0
    correct=0
    FPR=0
    y_test_flat = y_test.values.ravel()

    y_pred = []
    y_test = []
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)

            # Spłaszczenie wyjść modelu do jednowymiarowego wektora
            predicted = np.where(outputs.view(-1).numpy() < 0.5, 0, 1)
            # Konwersja y do płaskiej postaci
            y = y.view(-1).numpy()
            y_pred.append(predicted)
            y_test.append(y)
            
            total += y.size
            correct += (predicted == y).sum()


    #print(f'Accuracy of the network: {100* correct // total}%')

    y_pred = list(itertools.chain(*y_pred))
    y_test = list(itertools.chain(*y_test))
    
    y_pred = np.array(y_pred).flatten()
    y_test = np.array(y_test).flatten()

    accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    precision = precision_score(y_test, (y_pred > 0.5).astype(int))
    recall = recall_score(y_test, (y_pred > 0.5).astype(int))

    TP = np.sum((y_test == 1) & (y_pred > 0.5))
    TN = np.sum((y_test == 0) & (y_pred <= 0.5))
    FP = np.sum((y_test == 0) & (y_pred > 0.5))
    FN = np.sum((y_test == 1) & (y_pred <= 0.5))


    if FP + TN > 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0 

    #roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred) 
    cf_matrix = confusion_matrix(y_test, y_pred>0.5)

    # Wrap the model with the scikit-learn wrapper
    sklearn_model = SklearnPyTorchWrapper(model)

    # Feature Importance with eli5
    perm = PermutationImportance(sklearn_model, n_iter=5, random_state=42).fit(X_test, y_test_flat)
    df_fi = pd.DataFrame(dict(feature_names=X_test.columns.tolist(),
                          feat_imp=perm.feature_importances_, 
                          std=perm.feature_importances_std_,
                            ))
    df_fi = df_fi.round(4)
    #df_fi.sort_values('feat_imp', ascending=False)
    #save to csv
    #df_fi.to_csv('importance.csv', sep='\t', index=False)

    weights=eli5.show_weights(perm, feature_names=[f'feature_{i}' for i in range(X_test.shape[1])])
    #print(eli5.format_as_text(perm, feature_names=[f'feature_{i}' for i in range(X_test.shape[1])]))

    return accuracy, precision, recall, FPR, cf_matrix, fpr, tpr, df_fi
