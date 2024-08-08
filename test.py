from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

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

    return accuracy, precision, recall, cm
