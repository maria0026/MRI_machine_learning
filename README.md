# MRI_machine_learning
File Structure

The project folder contains four main scripts with "main" in their names, where the main logic of the program is executed. These files import scripts from the 'utils' folder. The argparser with default arguments is used in all four scripts. In all four scripts, there is a data_type argument, which is used to select data based on the norm_confirmed column: "positive" for only 1, "negative" for only 0, and "all" values.

    main_prepare.py: Prepares the data by replacing commas with periods, unifying delimiters, converting end-of-line characters, removing redundant columns, and merging CSV files. The processed files are by default saved in the folder data/data_type_norm_confirmed.
    main_delete_unnormal_features.py: Detects and removes columns that do not have a normal distribution. If testing on a different data type is required later, the argument test_data_type (default: None) should also be set to one of the values similarly to how data_type is set. The processed files are by default saved in the folder data/data_type_norm_confirmed_normal.
    main_age.py: Standardizes the data, divides the data by Estimated_Total_Intracranial_Volume, applies PCA, selects the gender, and trains and tests the model to predict age. An additional important argument is model_name.
    main_sex.py: Standardizes the data, applies PCA, and trains and tests the model to predict gender. An additional important argument is model_name.
    data: Folder containing original and processed data.
    results: Folder for saving CSV files with normality test results, outlier counts, PCA results, or component importance.
    data_info.ipynb: Jupyter notebook for generating plots.
    plots: Folder for saving plots.
    utils: Folder containing scripts with functions used in the main files (anomalies_detection.py, dimensions_reduction.py, nn_data.py, nn_model.py, plots.py, prepare_csv.py, prepare_dataset.py, test.py, train.py, volume_analysis.py).

The scripts should be run after activating the virtual environment with source .venv/bin/activate due to specific versions of imported libraries.

Library versions:

    numpy 1.26.4
    scipy 1.13.1
