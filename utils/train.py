from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import svm
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import nn_data, nn_model


def random_forest_model(X_train, y_train, param_dist, feature):
    rf = RandomForestClassifier(criterion='gini', max_features='sqrt')
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=10, 
                                    cv=5)
    
    rand_search.fit(X_train, y_train[feature])

    return rand_search

def random_forrest_regression_model(X_train, y_train, param_dist, feature):
    rf = RandomForestRegressor()
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=10, 
                                    cv=5)

    rand_search.fit(X_train, y_train[feature])

    return rand_search

def svm_classification_model(X_train, y_train, param_grid):
    clf = svm.SVC(probability=True)
    grid = RandomizedSearchCV(clf, param_distributions= param_grid, n_iter=10, cv=5) 
    grid.fit(X_train, y_train.values.ravel())

    return grid

def svm_regression_model(X_train, y_train, param_grid):
    clf = svm.SVR()
    grid = RandomizedSearchCV(clf, param_distributions= param_grid, n_iter=10, cv=5) 
    grid.fit(X_train, y_train.values.ravel())

    return grid


def feed_forward_neural_network(train_dataloader, input_dim, hidden_dim, output_dim, learning_rate, loss_fn, num_epochs, momentum=0, weight_decay=0):

    model = nn_model.NeuralNetwork(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)    
    print(model)


    loss_values = []
    model.train()
    for epoch in range(num_epochs):
        for X, y in train_dataloader:

            optimizer.zero_grad()
            pred = model(X)

            y = y.view(-1, 1)
            loss = loss_fn(pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
    '''
    step = np.linspace(0, 100, 2600)

    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    '''
    print("Training Complete")

    return model

def recurrent_neural_network(train_dataloader, seq_dim, input_dim, hidden_dim, layer_dim, output_dim, learning_rate, loss_fn, num_epochs, weight_decay=0):

    model = nn_model.RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  
    print(model)
    
    loss_list = []
    iteration_list = []
    count = 0
    model.train()
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_dataloader):

            train  = Variable(features.view(-1, seq_dim, input_dim))
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(train)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            count += 1
            
            if count % 250 == 0:        
                loss_list.append(loss.data)
                iteration_list.append(count)
                if count % 500 == 0:
                    print('Iteration: {}  Loss: {} '.format(count, loss.item()))

    return model
