import pandas as pd
import os 
from sklearn.model_selection import train_test_split

class FileProcessor:
    def __init__(self, path):
        self.path = path

    def replace_comma_with_dot(self, filename):
        file_path=os.path.join(self.path, filename)
        try:
            if not os.path.isfile(file_path):
                print(f"Plik nie istnieje: {file_path}")
                return

            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                content = file.read()

            if ',' in content:
                content = content.replace(',', '.')
                
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    file.write(content)
                print(f"Konwersja zakończona: {file_path}")
            else:
                print(f"Plik nie zawiera przecinków: {file_path}")

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania pliku {file_path}: {e}")

    def convert_delimiter(self, delimiter, old_delimiter, old_delimiter_2):

        files=os.listdir(self.path)
        print(files)
        for file in files:
            file_path = os.path.join(self.path, file)
            print(file_path)
            try:
                if not os.path.isfile(file_path):
                    print(f"Plik nie istnieje: {file_path}")
                    return

                with open(file_path, 'r', newline='', encoding='utf-8') as file:
                    content = file.read()
                
                if old_delimiter in content:
                    content = content.replace(old_delimiter, delimiter)
                    with open(file_path, 'w', newline='', encoding='utf-8') as file:
                        file.write(content)
                    print(f"delimiter zmieniony: {file_path}")

                elif old_delimiter_2 in content:
                    content = content.replace(old_delimiter_2, delimiter)
                    
                    with open(file_path, 'w', newline='', encoding='utf-8') as file:
                        file.write(content)
                    print(f"delimiter zmieniony: {file_path}")
                else:
                    print(f"Plik ma już dobry delimiter {file_path}")

            except Exception as e:
                print(f"Wystąpił błąd podczas przetwarzania pliku {file_path}: {e}")

    def delete_specified_columns(self, filenames, columns):
        for filename in filenames:
            filename= os.path.join(self.path, filename)
            df = pd.read_csv(filename, sep='\t')
            for column in columns:
                if column in df.columns:
                    df.drop(columns=column, inplace=True)
            
            df.to_csv(filename, sep="\t", index=False)

    def add_hemisphere_name(self, filenames, columns):
        for filename in filenames:
            path= os.path.join(self.path, filename)
            df = pd.read_csv(path, sep='\t')
            for column in columns:
                if column in df.columns:

                    df = df.rename(columns={column: filename[0:2]+'_'+column})

            print(df.columns)
            df.to_csv(path, sep="\t", index=False)

    def convert_line_endings(self):
        files=os.listdir(self.path)
        for file in files:
            file_path = os.path.join(self.path, file)
            
            try:
                if not os.path.isfile(file_path):
                    print(f"Plik nie istnieje: {file_path}")
                    return

                with open(file_path, 'r', newline='', encoding='utf-8') as file:
                    content = file.read()
                
                if '\r\n' in content:
                    content = content.replace('\r\n', '\n')
                    
                    with open(file_path, 'w', newline='', encoding='utf-8') as file:
                        file.write(content)
                    print(f"Konwersja zakończona: {file_path}")
                else:
                    print(f"Plik już jest w formacie Unix: {file_path}")

            except Exception as e:
                print(f"Wystąpił błąd podczas przetwarzania pliku {file_path}: {e}")



    def get_indexes_for_cleaning_dataset(self, filename, data_files=True, norm_confirmed=1):
        filepath = os.path.join(self.path, filename)
        df = pd.read_csv(filepath, sep='\t')
        
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
            #df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            #indexy zawierające 0 w kolumunie norm_confirmed
            zero_norm_confirmed = df[df['norm_confirmed']==1-norm_confirmed]
            zero_norm_confirmed_indexes = zero_norm_confirmed.index.tolist()

            #dostjemy indexy duplikatów poza pierwszym wystąpieniem
            if (norm_confirmed==1 or norm_confirmed==0):
                # Filtrujemy wiersze, gdzie 'norm_confirmed' = 1
                filtered_df = df[df['norm_confirmed'] == norm_confirmed]
                
                duplicates = filtered_df[filtered_df.duplicated(subset='identifier', keep='first')]
                duplicates_indexes = duplicates.index.tolist()
            else:
                duplicates = df[df.duplicated(subset='identifier', keep='first')]
                duplicates_indexes = duplicates.index.tolist()

            #concat all lists
            all_indexes = duplicates_indexes + missing_data_indexes + zero_norm_confirmed_indexes
        else:
            all_indexes = missing_data_indexes

        return all_indexes


    def clean_datasets(self, indexes_to_drop, folder_out):

        files=os.listdir(self.path)

        for filename in files:
            
            path= os.path.join(self.path, filename)
            print("Plik",path)
            df = pd.read_csv(path, sep='\t')
            df=df.drop(indexes_to_drop, inplace=False)
            print("rozmiar",df.shape)
            #save df to csv with delimiter ;
            df=df.dropna(axis=1, how='all')
            #check if folder_out exists
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
            df.to_csv(f'{folder_out}/{filename}', sep="\t", index=False)


    def concatenate_datasets(self, path):

        files=os.listdir(path)
        rozmiar=0
        dfs=[]
        for file in files:
            print(file)
            if not 'all_concatenated' in file:
                filepath= os.path.join(path, file)
                print(filepath)
                df = pd.read_csv(filepath, sep='\t')
                dfs.append(df)
                rozmiar+=df.shape[1]
                
        df = pd.concat(dfs, axis=1 )
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        df=df.dropna(axis=1, how='all')
        return df
        

