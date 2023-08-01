
import argparse
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import model as m
import math
from torch.utils.data import DataLoader
from train import train
from model import Prob_Type
from data.num_ds import num_ds as ds # change to universal dataset later

# ex:
# --prob_type c --num_in 22 --num_class 3 --csv_path data/pd_1/patient_data.csv
# use parser to get arguments -problem_type, -num_in, -num_classes (save as num_out, default to 1), -csv_path
def parse_args(main):
    def _wrapper():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--prob_type', choices=['b', 'c', 'r'], type=str, help='b (binary classification), c (classification), or r (regression)', required=True)
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
    if args.prob_type == 'b':
        print("You've indicated that you want to solve a binary classification problem.")
    elif args.prob_type == 'c':
        print("You've indicated that you want to solve a multi-class classification problem.")
    elif args.prob_type == 'r':
        print("You've indicated that you want to solve a regression problem.")
    print(f"You have {args.num_in} input features.")
    if args.prob_type == 'c':
        print(f"You've indicated that you have {args.num_out} output classes.")
    print(f"And your data is stored in {args.csv_path}.")
    print("Let's get started!")

    # make dataset object
    print("Loading data...")
    dataset = ds('data/pd_1/patient_data.csv')

    # Define the model
    print("Initializing model...")
    prob_type_converter = {'b': Prob_Type.B, 'c': Prob_Type.C, 'r': Prob_Type.R}

    # Define the loss
    if args.prob_type == 'b':
        criterion = nn.BCELoss()
    elif args.prob_type == 'c':
        criterion = nn.CrossEntropyLoss()  # use CrossEntropyLoss for multi-class classification
    elif args.prob_type == 'r':
        criterion = nn.MSELoss()


    # do a train with default values
    train(prob_type_converter[args.prob_type], dataset, criterion, args.num_in, args.num_out)

    # do a sweep to find optimal hyperparameters

    # save best model

    # loop to demo final model


main()