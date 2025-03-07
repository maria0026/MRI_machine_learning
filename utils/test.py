from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from utils import nn_data, nn_model
from eli5.sklearn import PermutationImportance
from torch.autograd import Variable


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
        'Predicted': y_pred.flatten(),
        'identifier': y_test['identifier'].values
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


def svm_regression_model(X_test, y_test, clf, z=None, feature=None, comp=True):
    print('Best hyperparameters:',  clf.best_params_)

    y_pred = clf.predict(X_test)
    #detrend the results
    if z is not None:
        y_pred = y_pred - z[0]*y_test[feature].values.ravel()**2 - z[1]*y_test[feature].values.ravel() - z[2]

    mse=mean_squared_error(y_test[feature], y_pred)
    rmse = float(format(np.sqrt(mean_squared_error(y_test[feature], y_pred)), '.3f'))
    mae = mean_absolute_error(y_test[feature], y_pred)

    y_test_flat = y_test[feature].values.ravel() 
    results_df = pd.DataFrame({
        'Actual': y_test_flat,  
        'Predicted': y_pred,
        'identifier': y_test['identifier'].values
    })

    # Feature Importance with eli5
    
    perm = PermutationImportance(clf.best_estimator_, scoring='neg_mean_absolute_error', n_iter=5).fit(X_test, y_test_flat)
    if comp:
        df_fi = pd.DataFrame(dict(component_names=X_test.columns.tolist(),
                            comp_imp=perm.feature_importances_, 
                            std=perm.feature_importances_std_,
                                ))
    else:
        df_fi = pd.DataFrame(dict(feature_name=X_test.columns.tolist(),
                            feature_importance=perm.feature_importances_, 
                            ))
    df_fi = df_fi.round(4)

    return mse, rmse, mae, results_df, df_fi

def svm_regression_model_quantiles(results_df, y_test, z_quantiles=None, feature=None, plot=False, first_quantile=None, last_quantile=None):
    y_pred=results_df['Predicted']

    if z_quantiles is not None:
        for quantile, y_pred_quant in z_quantiles.items():
            plt.plot(y_test[feature], y_pred_quant[0]*y_test[feature]+y_pred_quant[1], label=f"Quantile: {quantile}")

        if plot:
            plt.plot(y_test[feature], y_pred, 'o', color='b', alpha=0.5, label='Predicted')
            plt.xlabel("Actual age")
            plt.ylabel("Predicted age")
            plt.savefig("plots/quantiles_test.png")
            plt.show()
            
        identifiers_lower=[]
        identifiers_upper=[]
        sex_lower=[]
        sex_upper=[]
        y_pred_df=pd.DataFrame({'Predicted':np.array(y_pred)}, index=y_test.index)
    

        for i in range(len(y_pred_df)):
            prediction = y_pred_df['Predicted'].iloc[i]
            actual_value = np.array((y_test[feature]).iloc[i])
            
            if prediction < z_quantiles[first_quantile][0] * actual_value + z_quantiles[first_quantile][1]:
                identifiers_lower.append(y_test.loc[y_test.index[i], 'identifier'])
                sex_lower.append(y_test.loc[y_test.index[i], 'male'])

            if prediction > z_quantiles[last_quantile][0] * actual_value + z_quantiles[last_quantile][1]:
                identifiers_upper.append(y_test.loc[y_test.index[i], 'identifier'])
                sex_upper.append(y_test.loc[y_test.index[i], 'male'])


        return identifiers_lower, identifiers_upper, sex_lower, sex_upper

    else:
        return None, None, None, None



def neural_network_classification(X_test, y_test, batch_size, model, feature):
    test_data =nn_data.Data(X_test, y_test[feature])
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
    sklearn_model = nn_model.SklearnPyTorchWrapper(model)

    # Feature Importance with eli5
    perm = PermutationImportance(sklearn_model, n_iter=5).fit(X_test, y_test_flat)
    df_fi = pd.DataFrame(dict(component_names=X_test.columns.tolist(),
                          comp_imp=perm.feature_importances_, 
                          std=perm.feature_importances_std_,
                            ))
    df_fi = df_fi.round(4)


    return accuracy, precision, recall, FPR, cf_matrix, fpr, tpr, df_fi

def neural_network_regression(X_test, y_test, batch_size, model, feature):
    test_data =nn_data.Data(X_test, y_test[feature], y_test['identifier'].values)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    y_pred = []
    y_test = []

    model.eval()
    with torch.no_grad():
        for X, y, ids in test_dataloader:
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
    sklearn_model = nn_model.SklearnPyTorchWrapper(model)

    # Feature Importance with eli5
    perm = PermutationImportance(sklearn_model, scoring='neg_mean_absolute_error', n_iter=5).fit(X_test, y_test_flat)
    df_fi = pd.DataFrame(dict(component_names=X_test.columns.tolist(),
                          comp_imp=perm.feature_importances_, 
                          std=perm.feature_importances_std_,
                            ))
    df_fi = df_fi.round(4)

    return mse, rmse, mae, y_pred_df, df_fi

def recurrent_neural_network_classification(X_test, y_test, batch_size, seq_dim, input_dim, model, feature):
    test_data =nn_data.DataRNN(X_test, y_test[feature], sequence_length=seq_dim)
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


def recurrent_neural_network_regression(X_test, y_test, batch_size, seq_dim, input_dim, model, feature):
    test_data =nn_data.DataRNN(X_test, y_test[feature], y_test['identifier'].values, sequence_length=seq_dim)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    y_pred = []
    y_test_list = []
    identifiers_list = []
    model.eval()
    for features, y, identifiers in test_dataloader:
        features = Variable(features.view(-1, seq_dim, input_dim))
        
        outputs = model(features)
        predicted = outputs.view(-1).detach().numpy()
        # Konwersja y do płaskiej postaci
        y = y.view(-1).numpy()
        y_pred.append(predicted)
        y_test_list.append(y)
        identifiers_list.append(identifiers)

    y_pred_flat = np.array([item for sublist in y_pred for item in sublist])*100
    y_test_flat = np.array([item for sublist in y_test_list for item in sublist])*100
    identifiers_list = np.array([item for sublist in identifiers_list for item in sublist])
    #create dataframe from y_pred and y_test_list
    y_pred_df=pd.DataFrame({'Actual':np.array(y_test_flat),
                           'Predicted':np.array(y_pred_flat),
                           'identifier': identifiers_list})
    

    mse=mean_squared_error(y_test_flat, y_pred_flat)
    rmse = float(format(np.sqrt(mean_squared_error(y_test_flat, y_pred_flat)), '.3f'))
    mae = mean_absolute_error(y_test_flat, y_pred_flat)


    return mse, rmse, mae, y_pred_df