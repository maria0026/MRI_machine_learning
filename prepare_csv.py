import pandas as pd
import os 
from sklearn.model_selection import train_test_split

def replace_comma_with_dot(file_path):
    try:
        # Sprawdź, czy plik istnieje
        if not os.path.isfile(file_path):
            print(f"Plik nie istnieje: {file_path}")
            return

        # Odczytaj plik
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            content = file.read()
        
        # Sprawdź, czy potrzebna jest konwersja
        if ',' in content:
            # Zamień przecinki na kropki
            content = content.replace(',', '.')
            
            # Zapisz plik z poprawionymi przecinkami
            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                file.write(content)
            print(f"Konwersja zakończona: {file_path}")
        else:
            print(f"Plik nie zawiera przecinków: {file_path}")

    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania pliku {file_path}: {e}")

def convert_deliminer(folder, deliminer, old_deliminer, old_deliminer_2):

    files=os.listdir(folder)
    print(files)
    for file in files:
        file_path = os.path.join(folder, file)
        print(file_path)
        try:
            # Sprawdź, czy plik istnieje
            if not os.path.isfile(file_path):
                print(f"Plik nie istnieje: {file_path}")
                return

            # Odczytaj plik
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                content = file.read()
            
            # Sprawdź, czy potrzebna jest konwersja
            if old_deliminer in content:
                # Zamień końce linii z Windows na Unix
                content = content.replace(old_deliminer, deliminer)
                # Zapisz plik z poprawionymi końcami linii
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    file.write(content)
                print(f"Deliminer zmieniony: {file_path}")

            elif old_deliminer_2 in content:
                content = content.replace(old_deliminer_2, deliminer)
                
                # Zapisz plik z poprawionymi końcami linii
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    file.write(content)
                print(f"Deliminer zmieniony: {file_path}")
            else:
                print(f"Plik ma już dobry deliminer {file_path}")

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania pliku {file_path}: {e}")

def delete_specified_columns(folder, filenames, columns):
    for filename in filenames:
        filename= os.path.join(folder, filename)
        df = pd.read_csv(filename, sep='\t')
        for column in columns:
            if column in df.columns:
                df.drop(columns=column, inplace=True)
        
        df.to_csv(filename, sep="\t", index=False)

def add_hemisphere_name(folder, filenames, columns):
    for filename in filenames:
        path= os.path.join(folder, filename)
        df = pd.read_csv(path, sep='\t')
        for column in columns:
            if column in df.columns:

                df = df.rename(columns={column: filename[0:2]+'_'+column})

        print(df.columns)
        df.to_csv(path, sep="\t", index=False)

def convert_line_endings(folder):
    files=os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        
        try:
            # Sprawdź, czy plik istnieje
            if not os.path.isfile(file_path):
                print(f"Plik nie istnieje: {file_path}")
                return

            # Odczytaj plik
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                content = file.read()
            
            # Sprawdź, czy potrzebna jest konwersja
            if '\r\n' in content:
                # Zamień końce linii z Windows na Unix
                content = content.replace('\r\n', '\n')
                
                # Zapisz plik z poprawionymi końcami linii
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    file.write(content)
                print(f"Konwersja zakończona: {file_path}")
            else:
                print(f"Plik już jest w formacie Unix: {file_path}")

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania pliku {file_path}: {e}")



def get_indexes_for_cleaning_dataset(folder, filename, data_files=True):

    filename= os.path.join(folder, filename)
    df = pd.read_csv(filename, sep='\t')
    df.columns = df.columns.str.strip()  # Usuwa nadmiarowe białe znaki
    if data_files==True:
        #delete last columns
        #df=df.drop(df.columns[-1], axis=1, inplace=False)
        #delete columns with Unnamed
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    #indexy z brakującymi danymi
    missing_data = df[df.isnull().any(axis=1)]
    missing_data_indexes = missing_data.index.tolist()
  
    if data_files==False:

        #indexy zawierające 0 w kolumunie norm_confirmed
        zero_norm_confirmed = df[df['norm_confirmed']==0]
        zero_norm_confirmed_indexes = zero_norm_confirmed.index.tolist()

        #dostjemy indexy duplikatów poza pierwszym wystąpieniem
        # Filtrujemy wiersze, gdzie 'norm_confirmed' = 1
        filtered_df = df[df['norm_confirmed'] == 1]
        
        duplicates = filtered_df[filtered_df.duplicated(subset='identifier', keep='first')]
        duplicates_indexes = duplicates.index.tolist()

        #concat all lists
        all_indexes = duplicates_indexes + missing_data_indexes + zero_norm_confirmed_indexes
    else:
        all_indexes = missing_data_indexes

    return all_indexes



def clean_datasets(indexes_to_drop, folder, folder_out):

    files=os.listdir(folder)

    for filename in files:
        
        path= os.path.join(folder, filename)
        print("PLik",path)
        df = pd.read_csv(path, sep='\t')
        df=df.drop(indexes_to_drop, inplace=False)
        #save df to csv with deliminer ;
        df.to_csv(f'{folder_out}/{filename}', sep="\t", index=False)


def concatenate_datasets(folder):

    files=os.listdir(folder)
    rozmiar=0
    dfs=[]
    for file in files:
        print(file)
        if not 'all_concatenated' in file:
            path= os.path.join(folder, file)
           
            print(path)
            df = pd.read_csv(path, sep='\t')
            dfs.append(df)
            rozmiar+=df.shape[1]
    print(rozmiar)
            
    df = pd.concat(dfs, axis=1 )
    #df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    df=df.dropna(axis=1, how='all')
    #save df to csv
    df.to_csv(f'{folder}/all_concatenated.csv', sep='\t', index=False)

