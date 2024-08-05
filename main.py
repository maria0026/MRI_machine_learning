import prepare_df
import anomalies_detection
import pandas as pd
import os

path = 'data_4/original/Subjects.csv'
prepare_df.replace_comma_with_dot(path)

folder='data_4/original'

#prepare_df.merge_by_atlas(folder, 'A2009')
#prepare_df.merge_by_atlas(folder, 'APARC')
#prepare_df.merge_by_atlas(folder, 'DKT')

old_deliminer = ','
old_deliminer_2 = ';'
deliminer = '\t'

prepare_df.convert_deliminer(folder, deliminer, old_deliminer, old_deliminer_2)
prepare_df.convert_line_endings(folder)
columns=['Brain_Segmentation_Volume','Brain_Segmentation_Volume_Without_Ventricles','Brain_Segmentation_Volume_Without_Ventricles_from_Surf','Total_cortical_gray_matter_volume','Supratentorial_volume','Supratentorial_volume.1', 'Estimated_Total_Intracranial_Volume','Brain Segmentation Volume',	'Brain Segmentation Volume Without Ventricles']
#z wszystkich oprócz 1
filenames=['LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv','ASEG.csv', 'BRAIN.csv', 'WM.csv']
prepare_df.delete_specified_columns(folder, filenames, columns)

columns=['White_Surface_Total_Area', 'Mean_Thickness']
filenames=['LHA2009.csv', 'LHAPARC.csv', 'RHA2009.csv', 'RHAPARC.csv']
prepare_df.delete_specified_columns(folder, filenames, columns)

columns=['Volume of ventricles and choroid plexus']
filenames=['ASEG.csv']
prepare_df.delete_specified_columns(folder, filenames, columns)

columns=['White_Surface_Total_Area', 'Mean_Thickness']
filenames=['LHDKT.csv', 'RHDKT.csv']
prepare_df.add_hemisphere_name(folder, filenames, columns)


#uwuwanie duplikatów subjectów, brakujących danych i danych z norm_confirmed=0 na podstwaie Subjects.csv
filename='Subjects.csv'
indexes=prepare_df.get_indexes_for_cleaning_dataset(folder, filename, data_files=False)
#print(indexes)


# Lista plików CSV
filenames = ['WM.csv', 'ASEG.csv', 'BRAIN.csv', 'LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv', 'RHDKT.csv']

for filename in filenames:
    indexes_df=prepare_df.get_indexes_for_cleaning_dataset(folder,filename, data_files=True)
    if len(indexes_df)>0:
        indexes.append(indexes_df)
    else:
        print(f'No missing data in {filename}')

#delete duplicates from indexes

indexes = list(set(indexes))
print(indexes)
#czyszczenie danych


folder_out='data_4/cleaned_data'
prepare_df.clean_datasets(indexes, folder, folder_out)


folder_name=folder_out
prepare_df.convert_line_endings(folder_name)
prepare_df.concatenate_datasets(folder_name)


filename='data_4/cleaned_data/all_concatenated.csv'
df=pd.read_csv(filename, sep='\t')
print(df)

#test normalnosci- bez kolumn typu płec 
columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female']
anomalies_detection.test_normality(folder, filename, columns_to_drop)



