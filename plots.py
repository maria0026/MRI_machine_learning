import matplotlib.pyplot as plt
import seaborn as sns

def plot_some_data(df):
    #test normalnosci- bez kolumn typu płec 
    columns_to_drop=['identifier', 'norm_confirmed', 'sex', 'male', 'female']
    #anomalies_detection.test_normality(filename, columns_to_drop)
    df=df.drop(columns=columns_to_drop)
    #plot histograms for first 4 columns
    plt.figure(figsize=(10, 10))
    for i, column in enumerate(df.columns[:9]):
        plt.subplot(3, 3, i+1)
        plt.hist(df[column])
        plt.title(column)
    plt.subplots_adjust(hspace=0.5) 
    plt.show()

    plt.figure(figsize=(10, 10))
    for i, column in enumerate(df.columns[:9]):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x=df[column])
        plt.title(column)
    plt.subplots_adjust(hspace=1) 
    plt.show()