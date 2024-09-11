from utils import prepare_csv

norm_confimed=3

if norm_confimed==1:
    type='positive'
elif norm_confimed==0:
    type='negative'
else:
    type='all'

folder_out=f'data/{type}_norm_confirmed'


path = 'data/original/Subjects.csv'
prepare_csv.replace_comma_with_dot(path)

folder='data/original'
old_delimiter = ','
old_delimiter_2 = ';'
delimiter = '\t'

prepare_csv.convert_delimiter(folder, delimiter, old_delimiter, old_delimiter_2)
prepare_csv.convert_line_endings(folder)
columns=['Brain_Segmentation_Volume','Brain_Segmentation_Volume_Without_Ventricles','Brain_Segmentation_Volume_Without_Ventricles_from_Surf','Total_cortical_gray_matter_volume','Supratentorial_volume','Supratentorial_volume.1', 'Estimated_Total_Intracranial_Volume','Brain Segmentation Volume',	'Brain Segmentation Volume Without Ventricles']
#z wszystkich oprócz 1
filenames=['LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv','ASEG.csv', 'BRAIN.csv', 'WM.csv']
prepare_csv.delete_specified_columns(folder, filenames, columns)

columns=['White_Surface_Total_Area', 'Mean_Thickness']
filenames=['LHA2009.csv', 'LHAPARC.csv', 'RHA2009.csv', 'RHAPARC.csv']
prepare_csv.delete_specified_columns(folder, filenames, columns)

columns=['Volume of ventricles and choroid plexus', 'Left hemisphere cortical gray matter volume','Right hemisphere cortical gray matter volume']
filenames=['ASEG.csv']
prepare_csv.delete_specified_columns(folder, filenames, columns)

columns=['Left_hemisphere_cerebral_white_matter_volume', 'Right_hemisphere_cerebral_white_matter_volume', 'Total_cerebral_white_matter_volume']
filenames=['BRAIN.csv']
prepare_csv.delete_specified_columns(folder, filenames, columns)

columns=['White_Surface_Total_Area', 'Mean_Thickness']
filenames=['LHDKT.csv', 'RHDKT.csv']
prepare_csv.add_hemisphere_name(folder, filenames, columns)

#uwuwanie duplikatów subjectów, brakujących danych i danych z norm_confirmed=0 na podstwaie Subjects.csv
filename='Subjects.csv'
indexes=prepare_csv.get_indexes_for_cleaning_dataset(folder, filename, data_files=False, norm_confirmed=norm_confimed)


filenames = ['WM.csv', 'ASEG.csv', 'BRAIN.csv', 'LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv', 'RHDKT.csv']

for filename in filenames:
    indexes_df=prepare_csv.get_indexes_for_cleaning_dataset(folder,filename, data_files=True, norm_confirmed=norm_confimed)
    if len(indexes_df)>0:
        indexes.append(indexes_df)
    else:
        print(f'No missing data in {filename}')

#delete duplicates from indexes

indexes = list(set(indexes))
print("ilosc", len(indexes))

#czyszczenie danych

prepare_csv.clean_datasets(indexes, folder, folder_out)

folder_name=folder_out
prepare_csv.convert_line_endings(folder_name)
prepare_csv.concatenate_datasets(folder_name)