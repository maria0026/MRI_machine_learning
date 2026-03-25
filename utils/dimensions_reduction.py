import os
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collections import Counter
from itertools import combinations
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

class DimensionsReductor:

    def __init__(self):
        print("DimensionsReductor initialized")

    
    def get_top_features(self, loadings, n_features, n_pcs):
        important_indices = [np.abs(loadings[i]).argsort()[-n_features:][::-1] for i in range(n_pcs)]
        return important_indices


    def principal_component_analysis(self, X_train, X_test, components_nr, n_features=3, X_val=None, validation=False):

        pca_mri = PCA(components_nr)
        #pca_mri = KernelPCA(
        #    n_components=55, kernel="rbf", gamma=None
        #)

        train_pca = pca_mri.fit_transform(X_train)
        test_pca = pca_mri.transform(X_test)
        if validation:
            val_pca = pca_mri.transform(X_val)

        print("Train pca: ", train_pca.shape)
        #PC_values = np.arange(pca_mri.n_components_) + 1
        #plt.plot(PC_values, pca_mri.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        #plt.title('Scree Plot')
        #plt.xlabel('Principal Component')
        #plt.ylabel('Variance Explained')
        #plt.show()

        
        explained_variance_ratio=pca_mri.explained_variance_ratio_
        formatted_explained_variance = [f"{num:.10f}" for num in explained_variance_ratio]

        #mówi jak cechy przykładają się do komponentów
        component_loadings = pca_mri.components_
        n_pcs= component_loadings.shape[0]
        

        important_indices=self.get_top_features(component_loadings, n_features, n_pcs)
        initial_feature_names = X_train.columns
        
        #get the names
        important_names= [initial_feature_names[important_indices[i]] for i in range(n_pcs)]
        important_values= [component_loadings[i][important_indices[i]] for i in range(n_pcs)]

        importance_df = pd.DataFrame({
        f'Feature {j+1} Name': [important_names[i][j] for i in range(n_pcs)] for j in range(n_features)})

        importance_df['Explained Variability'] = formatted_explained_variance
        
        for j in range(n_features):
            importance_df[f'Feature {j+1} Value'] = [important_values[i][j] for i in range(n_pcs)]
            
        importance_df.index = range(1, n_pcs + 1)

        # Dodanie pustego wiersza
        empty_row = pd.DataFrame([[np.nan] * importance_df.shape[1]], columns=importance_df.columns)
        importance_df = pd.concat([importance_df, empty_row], ignore_index=True)

        # Ustawienie indeksów
        importance_df.index = range(1, len(importance_df) + 1)
        
        #importance_df = None
        if validation:
            return pca_mri, train_pca, val_pca, test_pca,  importance_df

        return pca_mri, train_pca, None, test_pca, importance_df

    def calculate_correlation_matrices(self, folder, folder_out):
        files=os.listdir(folder)
        print(files)
        for file in files:
            if not 'all' in file and not 'Subjects' in file:
                path=os.path.join(folder, file)
                df=pd.read_csv(path, sep='\t')
                df=df.dropna(axis=1, how='all')
                corr_matrix=df.corr()
                corr_matrix.to_csv(f'{folder_out}/{file[:-4]}_correlation_matrix.csv', sep='\t', index=True)


    def pair_files(self, folder, folder_out):
        files = os.listdir(folder)
        
        files.remove('Subjects.csv')
        files.remove('all_concatenated.csv')
        for i, file1 in enumerate(files):
            for j, file2 in enumerate(files):
                if i < j:  # Para plików jest unikalna (np. (file1, file2) ale nie (file2, file1))
                    # Wczytanie danych z plików
                    path1 = os.path.join(folder, file1)
                    path2 = os.path.join(folder, file2)

                    df1 = pd.read_csv(path1, sep='\t')
                    df2 = pd.read_csv(path2, sep='\t')

                    # Scalanie danych wzdłuż osi kolumn
                    df = pd.concat([df1, df2], axis=1)
                    df = df.dropna(axis=1, how='all')  # Usuwanie kolumn z tylko brakującymi wartościami

                    # Nazwa pliku wynikowego z połączonych plików
                    output_filename = f'{folder_out}/{os.path.splitext(file1)[0]}_{os.path.splitext(file2)[0]}_merged.csv'

                    # Zapisz połączony DataFrame do pliku CSV
                    df.to_csv(output_filename, sep='\t', index=False)
                    print(f'Saved: {output_filename}')
                
    def cluster_correlations(self, correlations):
        plt.figure(figsize=(12,5))

        correlations = correlations.fillna(0)
        np.fill_diagonal(correlations.values, 1)


        dissimilarity = 1 - abs(correlations)
        Z = linkage(squareform(dissimilarity), 'complete')

        dendrogram(Z, labels=correlations.columns, orientation='top', 
                leaf_rotation=90);
        plt.show()

        threshold = 0.8
        labels = fcluster(Z, threshold, criterion='distance')

        # Mapowanie etykiet do kolumn
        label_to_columns = {}
        for label, column in zip(labels, correlations.columns):
            if label not in label_to_columns:
                label_to_columns[label] = [column]
            else:
                label_to_columns[label].append(column)

        # Redukcja cech - wybór jednej cechy z każdej grupy
        selected_features = []
        for label, columns in label_to_columns.items():
            # Możesz wybrać np. pierwszą cechę lub cechę z najmniejszą wariancją
            selected_features.append(columns[0])


        # Keep the indices to sort labels
        labels_order = np.argsort(labels)

        # Build a new dataframe with the sorted columns
        for idx, i in enumerate(correlations.columns[labels_order]):
            if idx == 0:
                clustered = pd.DataFrame(correlations[i])
            else:
                df_to_append = pd.DataFrame(correlations[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
        
        correlations = clustered.corr()

        return correlations, selected_features
    

    def hierarchical_feature_selection(self, X, y, trainer, tester, model, svm_param_dist, test_feature,
                                    n_groups=10, top_fraction=0.5, final_n_features=10,
                                    max_subset_size=3, random_state=42):
        
        rng = np.random.default_rng(random_state)
        feature_names = list(X.columns)
        current_features = feature_names.copy()

        while len(current_features) > final_n_features:
            rng.shuffle(current_features)
            grouped_features = np.array_split(current_features, n_groups)
            all_top_features = []

            for group in tqdm(grouped_features, desc="Processing groups"):
                feature_dict={}
                group = list(group)
                X_group = X[group]

                # Train/test split for internal evaluation
                X_train, X_val, y_train, y_val = train_test_split(X_group, y, test_size=0.2, random_state=random_state)

                # Przejdź przez wszystkie kombinacje cech w grupie
                all_combos = []
                for r in range(1, min(max_subset_size + 1, len(group) + 1)):
                    all_combos.extend(list(combinations(group, r)))

                for combo in tqdm(all_combos, desc=f"Group with {len(group)} features", leave=False):
                    X_sub = X_train[list(combo)]

                    if model == "svm":
                        clf = trainer.svm_regression_model(X_sub, y_train, svm_param_dist, test_feature)
                        mse, rmse, mae, results_df, _ = tester.svm_regression_model(
                            X_val[list(combo)], y_val, clf, z=None, feature=test_feature, comp=False, importance=False
                        )
                        feature_dict[tuple(combo)]=mae
                    

                sorted_dict = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
                n_combinations = int(len(sorted_dict) * top_fraction)
                
                top_features = sorted_dict[:n_combinations]
                all_top_features.extend([feature for combo, _ in top_features for feature in combo])

            n_features = int(len(current_features) * top_fraction)
            feature_counts = Counter(all_top_features)
            most_common_features = feature_counts.most_common(n_features)
            current_features = [feature for feature, count in most_common_features]

        return current_features

    def umap(self, X_train, X_test, X_val=None, validation=False, n_components=30):

        reducer = umap.UMAP(n_components = n_components)

        embedding = reducer.fit_transform(X_train)
        test = reducer.transform(X_test)
        val = reducer.transform(X_val) if validation else None

        return reducer, embedding, val, test


    def nested_crossvalidation(self, X_trainval, y_trainval, param_grid, feature):

        #y_trainval = y_trainval.values.ravel()
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        all_selected_features = []

        for outer_train_idx, outer_val_idx in outer_cv.split(X_trainval, y_trainval):
            X_outer_train, X_outer_val = X_trainval.iloc[outer_train_idx], X_trainval.iloc[outer_val_idx]
            y_outer_train, y_outer_val = y_trainval.iloc[outer_train_idx], y_trainval.iloc[outer_val_idx]

            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
            feature_scores = Counter()

            for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
                X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
                y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]

                # Przykładowa procedura oceny cech — wybieramy top 10 według ważności z RandomForest
                clf = svm.SVR()
                model = RandomizedSearchCV(clf, param_distributions = param_grid, n_iter=10, cv=5) 
                model.fit(X_inner_train, y_inner_train[feature].values.ravel())
                #importances = model.feature_importances_
                result = permutation_importance(
                model.best_estimator_,
                X_inner_val,
                y_inner_val,
                scoring='neg_mean_absolute_error',
                n_repeats=5,
                random_state=42
            )
                top_features = X_inner_train.columns[np.argsort(result.importances_mean)[-10:]]

                feature_scores.update(top_features)

            # Wybór cech, które najczęściej pojawiały się w top 10
            selected_features = [feat for feat, count in feature_scores.items() if count >= 2]
            all_selected_features.extend(selected_features)

        # Agregacja cech ze wszystkich foldów zewnętrznych
        final_feature_counts = Counter(all_selected_features)
        final_features = [feat for feat, count in final_feature_counts.items() if count >= 3]
        
        return final_features