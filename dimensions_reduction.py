import os
import pandas as pd

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
            

def principal_component_analysis(train_data, test_data):
    print("f")