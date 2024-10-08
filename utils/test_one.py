import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def svm_regression(clf, X_test, y_test, z=None, z_quantiles=None, feature=None, plot=False, first_quantile=None, last_quantile=None, nr_of_fold=0):
    
    y_pred = clf.predict(X_test)
    y_pred = y_pred -  z[0]*y_test[feature].values.ravel()**2 - z[1]*y_test[feature].values.ravel() - z[2]
    x_axis = np.arange(0, 100, 0.01)
    for quantile, y_pred_quant in z_quantiles.items():
        plt.plot(x_axis, y_pred_quant[0]*x_axis+y_pred_quant[1], label=f"Quantile: {quantile}")

    if plot:
        plt.plot(y_test[feature], y_pred, 'o', color='b', alpha=0.5, label='Predicted')
        plt.savefig(f'plots/quantiles_{nr_of_fold}.png')
        plt.show()
        
    identifiers_lower=[]
    identifiers_upper=[]

    y_pred_df=pd.DataFrame({'Predicted':np.array(y_pred), 'Actual': y_test[feature].values.flatten()}, index=y_test.index)


    for i in range(len(y_pred_df)):
        prediction = y_pred_df['Predicted'].iloc[i]
        actual_value = np.array((y_test[feature]).iloc[i])
        
        if prediction < z_quantiles[first_quantile][0] * actual_value + z_quantiles[first_quantile][1]:
            identifiers_lower.append(y_test.loc[y_test.index[i], 'identifier'])

        if prediction > z_quantiles[last_quantile][0] * actual_value + z_quantiles[last_quantile][1]:
            identifiers_upper.append(y_test.loc[y_test.index[i], 'identifier'])


    return identifiers_lower, identifiers_upper, y_pred_df

