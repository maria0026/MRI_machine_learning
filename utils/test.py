from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils import nn_data
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.autograd import Variable
import shap

def random_forest_model(X_test, y_test, feature, rf):
    print('Best hyperparameters:',  rf.best_params_)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test[feature], y_pred)
    precision = precision_score(y_test[feature], y_pred)
    recall = recall_score(y_test[feature], y_pred)
    cm = confusion_matrix(y_test[feature], y_pred)

    #roc curve
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[feature], y_pred_proba) 
    
    return accuracy, precision, recall, cm, fpr, tpr


def random_forest_regression_model(X_test, y_test, feature, rf):
    print('Best hyperparameters:',  rf.best_params_)

    y_pred = rf.predict(X_test)
    
    mse=mean_squared_error(y_test[feature], y_pred)
    rmse = float(format(np.sqrt(mean_squared_error(y_test[feature], y_pred)), '.3f'))
    mae = mean_absolute_error(y_test[feature], y_pred)

    print(y_test.shape, y_pred.shape)
    results_df = pd.DataFrame({
        'Actual': y_test[feature].values.flatten(), 
        'Predicted': y_pred.flatten()
    })

    return mse, rmse, mae, results_df


def svm_classification_model(X_test, y_test, clf):
    print('Best hyperparameters:',  clf.best_params_)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    #roc curve
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
    

    return accuracy, precision, recall, cm, fpr, tpr


def svm_regression_model(X_test, y_test, clf):
    print('Best hyperparameters:',  clf.best_params_)

    y_pred = clf.predict(X_test)

    mse=mean_squared_error(y_test, y_pred)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    mae = mean_absolute_error(y_test, y_pred)

    y_test_flat = y_test.values.ravel() 
    results_df = pd.DataFrame({
        'Actual': y_test_flat,  
        'Predicted': y_pred
    })
    
    return mse, rmse, mae, results_df


class SklearnPyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
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


def neural_network_classification(X_test, y_test, batch_size, model):
    test_data =nn_data.Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    y_test_flat= y_test.values.ravel()
    FPR=0
    y_pred = []
    y_pred_prob = []
    y_test = []

    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = np.where(outputs.view(-1).numpy() < 0.5, 0, 1)
            y = y.view(-1).numpy()
            y_pred.append(predicted)
            y_pred_prob.append(outputs.view(-1).numpy())
            y_test.append(y)

    y_pred = list(itertools.chain(*y_pred))
    y_pred_prob = list(itertools.chain(*y_pred_prob))
    y_test = list(itertools.chain(*y_test))
    
    y_pred = np.array(y_pred).flatten()
    y_pred_prob = np.array(y_pred_prob).flatten()
    y_test = np.array(y_test).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    TP = np.sum((y_test == 1) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))

    if FP + TN > 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0 

    #roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) 
    cf_matrix = confusion_matrix(y_test, y_pred>0.5)

    # Wrap the model with the scikit-learn wrapper
    sklearn_model = SklearnPyTorchWrapper(model)

    # Feature Importance with eli5
    perm = PermutationImportance(sklearn_model, n_iter=5).fit(X_test, y_test_flat)
    df_fi = pd.DataFrame(dict(component_names=X_test.columns.tolist(),
                          comp_imp=perm.feature_importances_, 
                          std=perm.feature_importances_std_,
                            ))
    df_fi = df_fi.round(4)


    return accuracy, precision, recall, FPR, cf_matrix, fpr, tpr, df_fi

def neural_network_regression(X_test, y_test, batch_size, model):
    test_data =nn_data.Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    y_pred = []
    y_test = []

    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            #Spłaszczenie
            predicted = outputs.view(-1).numpy()
            y = y.view(-1).numpy()

            y_pred.append(predicted)
            y_test.append(y)
            
    y_pred_flat = np.array([item for sublist in y_pred for item in sublist])*100
    y_test_flat = np.array([item for sublist in y_test for item in sublist])*100
    #create dataframe from y_pred and y_test
    y_pred_df=pd.DataFrame({'Actual':np.array(y_test_flat),
                           'Predicted':np.array(y_pred_flat)})
    
    
    mse=mean_squared_error(y_test_flat, y_pred_flat)
    rmse = float(format(np.sqrt(mean_squared_error(y_test_flat, y_pred_flat)), '.3f'))
    mae = mean_absolute_error(y_test_flat, y_pred_flat)

    #feature_importance
    # Wrap the model with the scikit-learn wrapper
    sklearn_model = SklearnPyTorchWrapper(model)

    # Feature Importance with eli5
    perm = PermutationImportance(sklearn_model, scoring='neg_mean_absolute_error', n_iter=5).fit(X_test, y_test_flat)
    df_fi = pd.DataFrame(dict(component_names=X_test.columns.tolist(),
                          comp_imp=perm.feature_importances_, 
                          std=perm.feature_importances_std_,
                            ))
    df_fi = df_fi.round(4)

    return mse, rmse, mae, y_pred_df, df_fi

def recurrent_neural_network_classification(X_test, y_test, batch_size, seq_dim, input_dim, model):
    test_data =nn_data.DataRNN(X_test, y_test, sequence_length=seq_dim)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    y_pred = []
    y_test = []

    model.eval()
    for features, y in test_dataloader:
        features = Variable(features.view(-1, seq_dim, input_dim))
        
        outputs = model(features)
        predicted = outputs.view(-1).detach().numpy()
        # Konwersja y do płaskiej postaci
        y_pred.append(predicted)
        y=y.view(-1).numpy()
        y_test.append(y)


    y_pred_flat = list(itertools.chain(*y_pred))
    y_test_flat = list(itertools.chain(*y_test))
    
    y_pred_flat = np.array(y_pred_flat).flatten()
    y_test_flat = np.array(y_test_flat).flatten()
    y_pred_binary = (y_pred_flat > 0.5).astype(int)

    accuracy = accuracy_score(y_test_flat, y_pred_binary)
    precision = precision_score(y_test_flat, y_pred_binary)
    recall = recall_score(y_test_flat, y_pred_binary)

    TP = np.sum((y_test_flat == 1) & (y_pred_flat > 0.5))
    TN = np.sum((y_test_flat == 0) & (y_pred_flat <= 0.5))
    FP = np.sum((y_test_flat == 0) & (y_pred_flat > 0.5))
    FN = np.sum((y_test_flat == 1) & (y_pred_flat <= 0.5))


    if FP + TN > 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0 

    #roc curve
    fpr, tpr, thresholds = roc_curve(y_test_flat, y_pred_flat) 
    cf_matrix = confusion_matrix(y_test_flat, y_pred_flat>0.5)
    
    return accuracy, precision, recall, FPR, cf_matrix, fpr, tpr


def recurrent_neural_network_regression(X_test, y_test, batch_size, seq_dim, input_dim, model):
    test_data =nn_data.DataRNN(X_test, y_test, sequence_length=seq_dim)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    
    y_pred = []
    y_test = []

    model.eval()
    for features, y in test_dataloader:
        features = Variable(features.view(-1, seq_dim, input_dim))
        
        outputs = model(features)
        predicted = outputs.view(-1).detach().numpy()
        # Konwersja y do płaskiej postaci
        y = y.view(-1).numpy()
        y_pred.append(predicted)
        y_test.append(y)

    y_pred_flat = np.array([item for sublist in y_pred for item in sublist])*100
    y_test_flat = np.array([item for sublist in y_test for item in sublist])*100
    #create dataframe from y_pred and y_test
    y_pred_df=pd.DataFrame({'Actual':np.array(y_test_flat),
                           'Predicted':np.array(y_pred_flat)})
    

    mse=mean_squared_error(y_test_flat, y_pred_flat)
    rmse = float(format(np.sqrt(mean_squared_error(y_test_flat, y_pred_flat)), '.3f'))
    mae = mean_absolute_error(y_test_flat, y_pred_flat)


    return mse, rmse, mae, y_pred_df