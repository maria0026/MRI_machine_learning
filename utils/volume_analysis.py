import pandas as pd

def correlation_matrix_total_volume(path, folder_out):

    df=pd.read_csv(path, sep='\t')
    df=df.dropna(axis=1, how='all')
    df=df.drop(columns=['identifier', 'norm_confirmed', 'sex', 'female', 'age'])
    # Wybierz kolumny zawierające słowo 'area'
    area_columns= [col for col in df.columns if 'Area' in col or 'Total_gray_matter_volume' in col or 'Total_cerebral_white_matter_volume' in col]
    #indexes for male
    df_male=df[df['male']==1]
    df_female=df[df['male']==0]


    # Kolumna docelowa
    target_column = 'Estimated_Total_Intracranial_Volume'

    # Oblicz korelację
    correlations = {}
    for col in area_columns:
        correlations[col] = df_male[[col, target_column]].corr().iloc[0, 1]
        correlations[col+'_female']= df_female[[col, target_column]].corr().iloc[0, 1]
        correlations[col+'_all']=df[[col, target_column]].corr().iloc[0, 1]


    # Konwersja wyników na DataFrame
    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation_with_Intracranial_Volume'])
    corr_df.to_csv(f'{folder_out}/volume_area_correlation_matrix.csv', sep='\t', index=True)