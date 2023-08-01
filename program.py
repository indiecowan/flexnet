
import argparse
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import model as m
import math
from torch.utils.data import DataLoader
import data.pd_1.pd1_dataset as ds # change to universal dataset later

# ex:
# --prob_type c --num_in 10 --num_class 3 --csv_path boogies.csv
# use parser to get arguments -problem_type, -num_in, -num_classes (save as num_out, default to 1), -csv_path
def parse_args(main):
    def _wrapper():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--prob_type', type=str, help='b (binary classification), c (classification), or r (regression)', required=True)
        parser.add_argument('--num_in', type=int, help='number of input features', required=True)

        # SAVING NUM CLASS AS NUM OUT
        parser.add_argument('--num_class', type=int, help='number of output classes', dest="num_out")
        parser.add_argument('--csv_path', type=str, help='path to csv file', required=True)

        args = parser.parse_args()

        # Manually enforce that num_class is required if prob_type is 'c'.
        if args.prob_type == 'c':
            if args.num_out is None:
                parser.error("--num_class is required when --prob_type is 'c'.")
        else:
            args.num_out = 1

        main(args)

    return _wrapper

# make main
@parse_args
def main(args):
    print(args)
    print("Welcome to flexnet!")
    print(f"You've indicated that you want to solve a {args.prob_type} problem.")
    print(f"You have {args.num_in} input features.")
    if args.prob_type == 'c':
        print(f"You've indicated that you have {args.num_out} output classes.")
    print(f"And your data is stored in {args.csv_path}.")
    print("Let's get started!")

    # make dataset object
    print("Loading data...")

    # do a train with default values

    # do a sweep to find optimal hyperparameters

    # save best model

    # loop to demo final model


main()