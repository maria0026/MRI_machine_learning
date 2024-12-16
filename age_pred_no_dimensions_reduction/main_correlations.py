import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
def main(args):

    df_gap_all=pd.DataFrame()
    for atlas in args.atlases:
        path=f'{args.results_directory}/{atlas}'
        files= os.listdir(path)

        for file in files:
            if file==f'test_{args.data_type}_regression_results_{args.model_name}_valid_{args.valid}.csv':
                df=pd.read_csv(f'{path}/{file}', sep='\t')
                list_actual = [column for column in df.columns if 'Actual' in column]
                list_predicted = [column for column in df.columns if 'Predicted' in column]
                list_gap=[]
                for i in range(len(list_actual)):
                    list_gap.append(df[list_predicted[i]].values-df[list_actual[i]].values)
                df_gap=pd.DataFrame(list_gap).T
                df_gap.columns=[f'{atlas}_{i}' for i in range(len(list_actual))]
                df_gap_all=pd.concat([df_gap_all, df_gap], axis=1)
    print(df_gap_all)
    corr=df_gap_all.corr(method='spearman')
    sns.heatmap(corr)
    plt.title(f'correlations of age gap between atlases')
    plt.show()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for age preidction")
    parser.add_argument("--data_type", nargs="?", default="positive", help="Type of dataset based on norm_confirmed: positive/negative/all", type=str)
    #parser.add_argument("--atlas", nargs="?", default="a2009", help="atlas", type=str)
    parser.add_argument("--atlases", nargs="?", default=["a2009", "APARC", 'DKT', "ASEG", "WM"], help="atlases", type=list)
    parser.add_argument("--model_name", nargs="?", default="svm", help="Model name: forest/svm/fnn/rnn", type=str)
    parser.add_argument("--test_one", nargs="?", default=0, help="Test one case", type=bool)
    parser.add_argument("--columns_to_drop", nargs="?", default=['identifier','norm_confirmed', 'sex', 'female'], help="Columns to drop", type=list)
    parser.add_argument("--division_by_total_volume", nargs="?", default=1, help="Divide volumetric data by Estimated_Total_Intracranial_Volume: 1/0", type=bool)
    parser.add_argument("--label_names", nargs="?", default=["age"], help="Predicted parameters, list", type=list)
    parser.add_argument("--valid", nargs="?", default=0, help="create valid set: 0/1", type=bool)
    parser.add_argument("--column_to_copy", nargs="?", default=['male'], help="Columns to copy", type=list)
    parser.add_argument("--first_quantile", nargs="?", default=0.01, help="First quantile for svm regression", type=float)
    parser.add_argument("--last_quantile", nargs="?", default=0.99, help="Last quantile for svm regression", type=float)
    parser.add_argument("--batch_size", nargs="?", default=64, help="Batch size", type=int)
    parser.add_argument("--rnn_seq_dim", nargs="?", default=1, help="Sequence dimension for recurrent neural network", type=int)
    parser.add_argument("--plot", nargs="?", default=1, help="Plot results", type=bool)
    parser.add_argument("--nr_of_train", nargs="?", default=5, help="Number of train dataset", type=int)
    parser.add_argument("--results_directory", nargs="?", default="results", help="Directory for results", type=str)
    args = parser.parse_args()
    main(args)    