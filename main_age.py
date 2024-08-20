import pandas as pd
import numpy as np
import dimensions_reduction
import prepare_dataset
import plots
import train
import test
from sklearn.metrics import roc_curve, auc


filename='data_4/normal/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')

#redukcja wymiarowości
#podział na zbiór treningowy i testowy
df=pd.read_csv('data_4/normal/all_concatenated.csv', sep='\t')
df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female'])
label_names=['age']
column_to_copy='male'

prepare_dataset.divide_by_total_volume(df)
X_train, X_test, y_train, y_test=prepare_dataset.split_dataset(df, label_names)
X_train_to_stardarize=X_train.drop(columns=column_to_copy)
X_test_to_stardarize=X_test.drop(columns=column_to_copy)
X_train_standarized, X_test_stantarized=prepare_dataset.standarize_data(X_train_to_stardarize, X_test_to_stardarize)
X_train=pd.concat([X_train_standarized, X_train[column_to_copy]], axis=1)
X_test=pd.concat([X_test_stantarized, X_test[column_to_copy]], axis=1)

#PCA
components_nr=40

pca_mri, train_pca, test_pca, importance_df=dimensions_reduction.principal_component_analysis(X_train, X_test, components_nr)
explained_variance_ratio=pca_mri.explained_variance_ratio_
formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]
print('Explained variability per principal component: {}'.format(formatted_explained_variance))

#mamy teraz train_pca i test_pca
train_principal_Df = pd.DataFrame(data = train_pca
            , columns = [str(i) for i in range(1,train_pca.shape[1]+1)], index=X_train.index)

test_principal_Df = pd.DataFrame(data = test_pca
            , columns = [str(i) for i in range(1,test_pca.shape[1]+1)], index=X_test.index)

column_to_copy='male'
train_principal_Df[column_to_copy]=X_train[column_to_copy]
test_principal_Df[column_to_copy]=X_test[column_to_copy]

X_train=train_principal_Df
X_test=test_principal_Df  

#2 osobne modele- dla mężczyzn i kobiet

X_train_male=train_principal_Df[train_principal_Df['male']==1]
train_indices=X_train_male.index.tolist()
y_train_male=y_train.loc[train_indices]

X_test_male=test_principal_Df[test_principal_Df['male']==1] 
test_indices=X_test_male.index.tolist()
y_test_male=y_test.loc[test_indices]

#female
X_train_female=train_principal_Df[train_principal_Df['male']==0]
train_indices=X_train_female.index.tolist()
y_train_female=y_train.loc[train_indices]

X_test_female=test_principal_Df[test_principal_Df['male']==0]
test_indices=X_test_female.index.tolist()
y_test_female=y_test.loc[test_indices]

X_train=train_principal_Df
X_test=test_principal_Df 

'''
X_train=X_train_male
y_train=y_train_male
X_test=X_test_male
y_test=y_test_male



X_train=X_train_female
y_train=y_train_female
X_test=X_test_female
y_test=y_test_female
'''

feature='age'
print("Odchylenie",np.std(y_train[feature]))
print("Srednia", np.mean(y_train[feature]))
rf=train.random_forrest_regression_model(X_train, y_train, feature)
best_rf = rf.best_estimator_
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances=feature_importances.drop('male')
feature_importances.index = feature_importances.index.astype(int)
print(feature_importances)
#sort ascending by indexes
feature_importances=feature_importances.sort_index()
print(feature_importances.index)
print("wart", feature_importances.values)
importance_df['component_importance_on_tree']=feature_importances.values
#save to csv
importance_df.to_csv('importance_age.csv', sep='\t', index=True)

mse, rmse, mae= test.random_forest_regression_model(X_test, y_test, feature, rf)
print("Mean squared error", mse)
print("Root mean squared error", rmse)
print("Mean absolute error", mae)

