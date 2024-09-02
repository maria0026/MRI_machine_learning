from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import svm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from utils import nn_data
from utils import nn_model
from torch.autograd import Variable

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

    param_grid = {'C': randint(30,40),  
              'gamma': uniform(0.0001,0.0005-0.0001), 
              'kernel': ['rbf']}  
  
    grid = RandomizedSearchCV(clf, param_distributions= param_grid, n_iter=10, cv=5) 
    #Train the model using the training sets
    grid.fit(X_train, y_train.values.ravel())

    return grid

def svm_regression_model(X_train, y_train):
    
    clf = svm.SVR()

    param_grid = {'C': randint(30,40),  
            'gamma': uniform(0.0001,0.0005-0.0001), 
            'kernel': ['rbf']}  

    grid = RandomizedSearchCV(clf, param_distributions= param_grid, n_iter=10, cv=5) 
    #Train the model using the training sets
    grid.fit(X_train, y_train.values.ravel())

    return grid



def layer_neural_network(X_train, y_train, input_dim, hidden_dim, output_dim, learning_rate, loss_fn, num_epochs):
    batch_size = 64
    #Instantiate training data
    train_data = nn_data.Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    model = nn_model.NeuralNetwork(input_dim, hidden_dim, output_dim)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)    
    loss_values = []

    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
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

def recurrent_neural_network(X_train, y_train, X_test, y_test, seq_dim, input_dim, hidden_dim, layer_dim, output_dim, learning_rate, loss_fn, num_epochs):
    batch_size = 64
    #Instantiate training data
    
    train_data = nn_data.DataRNN(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = nn_data.DataRNN(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    model = nn_model.RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    print(model)

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Example: Reduce learning rate
  
    loss_list = []
    iteration_list = []
    num_epochs = 100
    count = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):

            train  = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)
                
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(train)
            #print(outputs)
            
            # Calculate softmax and ross entropy loss
            loss = loss_fn(outputs, labels)
            
            # Calculating gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            
            # Update parameters
            optimizer.step()
            
            count += 1
            
            if count % 250 == 0:        
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                #accuracy_list.append(accuracy)
                if count % 500 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {} '.format(count, loss.item()))

    return model