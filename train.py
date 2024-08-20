from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import svm


def random_forest_model(X_train, y_train, feature):


    param_dist = {'n_estimators': randint(50,500),
                'max_depth': randint(1,35),
                'min_samples_leaf': randint(1,8),
                'min_samples_split': [2,3,5,8,10,14,20]}

    # Create a random forest classifier
    rf = RandomForestClassifier(criterion='gini', max_features='sqrt')

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=10, 
                                    cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train[feature])

    return rand_search

def random_forrest_regression_model(X_train, y_train, feature):

    param_dist = {'n_estimators': randint(50,500),
                'max_depth': randint(1,35),
                'min_samples_leaf': randint(1,8),
                'min_samples_split': [2,3,5,8,10,14,20]}
    
     # Create a random forest regressor
    rf = RandomForestRegressor()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=10, 
                                    cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train[feature])

    return rand_search

def svm_classification_model(X_train, y_train):

    #Create a svm Classifier
    clf = svm.SVC(probability=True)

    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
    grid = RandomizedSearchCV(clf, param_distributions= param_grid, n_iter=10, cv=5) 
    #Train the model using the training sets
    clf.fit(X_train, y_train)

    return clf


