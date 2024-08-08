from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split

def random_forest_model(X_train, y_train, feature):

    param_dist = {'n_estimators': randint(50,500),
                'max_depth': randint(1,20)}

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train[feature])

    return rand_search
