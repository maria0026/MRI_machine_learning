from utils import prepare_csv
import argparse

def main(args):
    processor = prepare_csv.FileProcessor(args.path)
    processor.change_column_names()

if __name__ == "__main__":  

    parser = argparse.ArgumentParser("parser for changing column names")  
    parser.add_argument("--path", nargs="?", default="data/pathology", help="Path to the folder where the original files are", type=str)
    args = parser.parse_args()
    main(args)
