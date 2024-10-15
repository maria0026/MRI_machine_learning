import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import het_white
from statsmodels.tools.tools import add_constant
from scipy.stats import skew, kurtosis
from scipy import stats

def calculate_trends(df, column, X_feature_name):

    X = df[X_feature_name].values
    ransac = RANSACRegressor()
    degree = 3
    model = make_pipeline(PolynomialFeatures(degree), ransac)
    
    y = df[column].values
    model.fit(X, y)
    ransac_model = model.named_steps['ransacregressor']

    # Współczynniki wielomianu (dla dopasowanego modelu)
    coefficients = ransac_model.estimator_.coef_
    intercept = ransac_model.estimator_.intercept_

    return model, coefficients, intercept


def white_test(df, column, X_feature_name, model):

    X = df[X_feature_name].values
    y = df[column].values
    #sort y by x
    L = sorted(zip(X,y))
    new_x, new_y = zip(*L)
    
    #white test
    residuals = new_y - model.predict(new_x)
    #add column of ones
    X_with_constant = add_constant(new_x)
    white_test = het_white(residuals,  X_with_constant)

    labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
    white_test = dict(zip(labels, white_test))

    return white_test['F-Test p-value']


def scores(df, column, X_feature_name, model):

    X = df[X_feature_name].values
    y = df[column].values
    #sort y by x
    L = sorted(zip(X,y))
    new_x, new_y = zip(*L)

    new_y = new_y - model.predict(new_x)
    mean = np.mean(new_y)
    std = np.std(new_y, ddof=1)
    skewness = skew(new_y)
    kurt = kurtosis(new_y)

    mad = stats.median_abs_deviation(new_y)
    median = np.median(y)

    q1 = np.quantile(new_y, 0.25)
    q3 = np.quantile(new_y, 0.75)
    iqr = q3 - q1
    min_value = np.min(new_y)
    max_value = np.max(new_y)

    return mean, std, skewness, kurt, mad, median, iqr, min_value, max_value


def calculate_quantiles(df, column, X_feature_name, quantiles, model):
    X = df[X_feature_name].values
    y = df[column].values
    #sort y by x
    L = sorted(zip(X,y))
    new_x, new_y = zip(*L)
    new_y = new_y - model.predict(new_x)
    quantiles_array = []
    indices_array = []

    for quantile in quantiles:
        lower_quantile=np.quantile(new_y, quantile)
        upper_quantile=np.quantile(new_y, 1-quantile)
        quantiles_array.append(lower_quantile)
        quantiles_array.append(upper_quantile)
        indices_array.append(f'{quantile} quantile')
        indices_array.append(f'{1-quantile} quantile')

    return quantiles_array, indices_array

    '''
    degree = 3
    ransac_model = model.named_steps['ransacregressor']
    coefficients = ransac_model.estimator_.coef_
    intercept = ransac_model.estimator_.intercept_
    x_range = np.linspace(5, 85, 1000).reshape(-1, 1)  # Przykładowe punkty do przewidywania
    #y2 = x_range**3 *coefficients[3]+ x_range**2 * coefficients[2] + x_range * coefficients[1] + intercept
    y_ransac = model.predict(x_range)
    # Rysowanie wyników
    plt.scatter(X, y, color='blue', label='Dane')
    #plt.plot(x_range, y_ransac, color='red', label=f'Wielomian stopnia {degree}')
    plt.scatter(new_x, new_y, color='red', label=f'po odjęciu wielomianu stopnia {degree}')
    #plt.plot(x_range, y2, color='green', label=f'Wielomian stopnia {degree}')
    plt.title(f'{column}')
    plt.legend()
    plt.show()
    '''
        


