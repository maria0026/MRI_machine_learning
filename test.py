from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

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