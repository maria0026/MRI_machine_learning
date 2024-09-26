import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
from sklearn.utils.fixes import parse_version, sp_version

def svm_regression_model(X_val, y_val, clf, feature, plot=False):

    y_pred = clf.predict(X_val)

    y_test_flat = y_val[feature].values.ravel() 
    results_df = pd.DataFrame({
        'Actual': y_test_flat,  
        'Predicted': y_pred
    })
    
    unique_actual, indices = np.unique(y_test_flat, return_index=True)
    mean_predicted = [np.mean([y_pred[j] for j in range(len(y_test_flat)) if y_test_flat[j] == ua]) for ua in unique_actual]
    mean_predicted_gap = [mean_predicted[i]- unique_actual[i] for i in range(len(unique_actual))]
    # Dopasowanie wielomianu do unikalnych danych
    z = np.polyfit(unique_actual, mean_predicted_gap, 2)
    print("Wielomian dopasowania", z[0], z[1], z[2])

    y_pred=y_pred-z[0]*y_val[feature].values.ravel()**2-z[1]*y_val[feature].values.ravel()-z[2]
    
    #transform df to numpy array
    X = y_val[feature].to_numpy()
    y=y_pred

    #sort both arrays
    idx = np.argsort(X[:, 0])
    X = X[idx]
    y = y[idx]

    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    predictions = {}
    out_bounds_predictions = np.zeros_like(y, dtype=np.bool_)
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver=solver)
        y_pred = qr.fit(X, y).predict(X)
        predictions[quantile] = y_pred

        if quantile == min(quantiles):
            out_bounds_predictions = np.logical_or(
                out_bounds_predictions, y_pred >= y
            )
        elif quantile == max(quantiles):
            out_bounds_predictions = np.logical_or(
                out_bounds_predictions, y_pred <= y
            )

    z_quantiles={}

    for quantile, y_pred in predictions.items():
        #fit function
        z_quantile = np.polyfit(X.ravel(), y_pred, 1)
        plt.plot(X, z_quantile[0]*X.ravel() + z_quantile[1], label=f"Quantile: {quantile}")
        z_quantiles[quantile]=z_quantile
        #plt.plot(X, y_pred, label=f"Quantile: {quantile}")
    if plot:
        plt.scatter(
            X[out_bounds_predictions],
            y[out_bounds_predictions],
            color="black",
            marker="+",
            alpha=0.5,
            label="Outside interval",
        )
        plt.scatter(
            X[~out_bounds_predictions],
            y[~out_bounds_predictions],
            color="black",
            alpha=0.5,
            label="Inside interval",
        )

        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        _ = plt.title("Quantiles of heteroscedastic Normal distributed target")
        plt.show()

    return z, z_quantiles