import argparse
from utils import prepare_csv
import pandas as pd
import os
import shutil

def main(args):
    
    processor = prepare_csv.FileProcessor(args.path)
    
    #processor.replace_comma_with_dot(args.subjects_file)
    processor.convert_delimiter(args.delimiter, args.old_delimiter, args.old_delimiter_2)
    processor.convert_line_endings()

    if args.data_type=="positive":
        norm_confimed=1
    elif args.data_type=="negative":
        norm_confimed=0
    elif args.data_type=="all":
        norm_confimed=3

    #deleting duplicated columns
    columns=['Brain_Segmentation_Volume','Brain_Segmentation_Volume_Without_Ventricles','Brain_Segmentation_Volume_Without_Ventricles_from_Surf','Total_cortical_gray_matter_volume','Supratentorial_volume','Supratentorial_volume.1', 'Estimated_Total_Intracranial_Volume','Brain Segmentation Volume',	'Brain Segmentation Volume Without Ventricles']
    #z wszystkich oprócz 1
    filenames=['LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv','ASEG.csv', 'BRAIN.csv', 'WM.csv']
    processor.delete_specified_columns(filenames, columns)

    columns=['White_Surface_Total_Area', 'Mean_Thickness']
    filenames=['LHA2009.csv', 'LHAPARC.csv', 'RHA2009.csv', 'RHAPARC.csv']
    processor.delete_specified_columns(filenames, columns)

    columns=['Left hemisphere cortical gray matter volume', 'Right hemisphere cortical gray matter volume','ASEG-Left hemisphere cerebral white matter volume','ASEG-Right hemisphere cerebral white matter volume', 'ASEG-Total cerebral white matter volume','ASEG-Subcortical gray matter volume', 'ASEG-Total gray matter volume', 'ASEG-Supratentorial volume', 'ASEG-Mask Volume', 'ASEG-Supratentorial volume_notvent', 'Supratentorial volume.1']
    filenames=['ASEG.csv']
    processor.delete_specified_columns(filenames, columns)

    columns=['Total_cerebral_white_matter_volume', 'Volume of ventricles and choroid plexus', 'Left_hemisphere_cerebral_white_matter_volume', 'Right_hemisphere_cerebral_white_matter_volume']
    filenames=['BRAIN.csv']
    processor.delete_specified_columns(filenames, columns)

    #for those files where these columms are, we add hemisphere name
    columns=['White_Surface_Total_Area', 'Mean_Thickness']
    filenames=['LHDKT.csv', 'RHDKT.csv']
    processor.add_hemisphere_name(filenames, columns)


    #searching for indexes of subjects with missing data, duplicates and norm_confired subset
    indexes=processor.get_indexes_for_cleaning_dataset(args.subjects_file, data_files=False, norm_confirmed=norm_confimed)
    filenames = ['WM.csv', 'ASEG.csv', 'BRAIN.csv', 'LHA2009.csv', 'LHAPARC.csv', 'LHDKT.csv', 'RHA2009.csv', 'RHAPARC.csv', 'RHDKT.csv']

    for filename in filenames:
        indexes_df=processor.get_indexes_for_cleaning_dataset(filename, data_files=True, norm_confirmed=norm_confimed)
        if os.path.isfile(filename):
            print("rozmiar", pd.read_csv(f'{args.path}/{filename}', sep='\t').shape)
            if len(indexes_df)>0:
                indexes.append(indexes_df)
            else:
                print(f'No missing data in {filename}')

    #delete duplicates from indexes
    indexes = list(set(indexes))
    print("ilosc", len(indexes))

    #clean the data based on indexes
    folder_out=f'data/{args.data_type}_norm_confirmed'
    if os.path.exists(folder_out):
        shutil.rmtree(folder_out)
        os.makedirs(folder_out)
    processor.clean_datasets(indexes, folder_out)

    folder_name=folder_out
    processor.convert_line_endings()
    concatenated_df=processor.concatenate_datasets(folder_name)
    print("concatenated_df", concatenated_df.shape, "to", f'{folder_name}/all_concatenated.csv')
    concatenated_df.to_csv(f'{folder_name}/all_concatenated.csv', sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for deleting unnormal features")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    parser.add_argument("--subjects_file", nargs="?", default="Subjects.csv", help="Name of the file with subjects", type=str)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier', 'norm_confirmed', 'sex', 'male', 'female', 'age','Estimated_Total_Intracranial_Volume'], help="Columns to drop", type=list)
    parser.add_argument("--path", nargs="?", default="data/original", help="Path to the folder where the original files are", type=str)
    parser.add_argument("--old_delimiter", nargs="?", default=",", help="Old delimiter", type=str)
    parser.add_argument("--old_delimiter_2", nargs="?", default=";", help="Old delimiter", type=str)
    parser.add_argument("--delimiter", nargs="?", default="\t", help="New delimiter", type=str)
    #parser.add_argument("--contatenated_filename", nargs="?", default="all_concatenated.csv", help="Name of the file with subjects", type=str)
    #parser.add_argument("--folder_out_end", nargs="?", default="norm_confirmed", help="End of folder out name", type=str)
    args = parser.parse_args()
    main(args)
