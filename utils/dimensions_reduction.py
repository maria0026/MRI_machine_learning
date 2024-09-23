import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.manifold import TSNE

class DimensionsReductor:

    def __init__(self):
        print("DimensionsReductor initialized")

    
    def get_top_features(self, loadings, n_features, n_pcs):
        important_indices = [np.abs(loadings[i]).argsort()[-n_features:][::-1] for i in range(n_pcs)]
        return important_indices


    def principal_component_analysis(self, X_train, X_test, components_nr, n_features=3, X_val=None, validation=False):

        pca_mri = PCA(components_nr)

        train_pca = pca_mri.fit_transform(X_train)
        test_pca = pca_mri.transform(X_test)
        if validation:
            val_pca = pca_mri.transform(X_val)

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

        # Dodanie indeksów jako numerów komponentów
        importance_df.index = range(1, n_pcs + 1)

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
    

    def stochastic_neighbor_embedding(self, X_train, X_test, components_nr):
        tsne = TSNE(n_components=2, random_state=42)
        train_tsne = tsne.fit_transform(X_train)
        #test_tsne=tsne.transform(X_test)
        test_tsne=X_test
        tsne.kl_divergence_

        return tsne, train_tsne, test_tsne
