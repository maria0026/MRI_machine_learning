import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns

def calculate_correlation_matrices(folder, folder_out):
    files=os.listdir(folder)
    for file in files:
        if not 'all' in file and not 'Subjects' in file:
            path=os.path.join(folder, file)
            df=pd.read_csv(path, sep='\t')
            df=df.dropna(axis=1, how='all')
            corr_matrix=df.corr()
            corr_matrix.to_csv(f'{folder_out}/{file[:-4]}_correlation_matrix.csv', sep='\t', index=True)


def pair_files(folder, folder_out):
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
            
def cluster_correlations(correlations):
    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, labels=correlations.columns, orientation='top', 
            leaf_rotation=90);
    plt.show()

    threshold = 0.8
    labels = fcluster(Z, threshold, criterion='distance')

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

    return correlations



def principal_component_analysis(X_train, X_test, components_nr):

    pca_mri = PCA(n_components=components_nr)

    train_pca = pca_mri.fit_transform(X_train)
    test_pca = pca_mri.transform(X_test)

    return pca_mri, train_pca, test_pca

    
