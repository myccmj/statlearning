import scipy.io as sio
import numpy as np
import torch
import argparse

from algs import Model
import train

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="spambase", choices=['spambase', 'gisette', 'madelon'])
    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--classifier", default="linear")                 
    parser.add_argument("--method", default="lagrange")
    parser.add_argument("--train_type", default="base")
    parser.add_argument("--split_rate", default=0.8, type=float)
    parser.add_argument("--bs_ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)             
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--file_name", default="")
    args = parser.parse_args()

    return args



def load_data(args):
    if args.dataset == 'spambase':
        data = sio.loadmat('./data/spambase.mat')
        x_data = data['X'].astype(np.float32)
        y_data = data['y'].astype(np.float32)
        ind_0 = np.reshape(data['y']==-1, -1)
        ind_1 = np.reshape(data['y']==1, -1)
    
    return x_data[ind_0, :], x_data[ind_1, :], y_data[ind_0, :], y_data[ind_1, :]



if __name__ == "__main__":

    args = config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_data_0, x_data_1, y_data_0, y_data_1 = load_data(args)
    model = Model(x_data_0.shape[1], args.delta, args.lr, args.classifier, args.method)
    if args.train_type == 'base':
        train_op = train.Base_train(model, x_data_0, x_data_1, y_data_0, y_data_1, args.epochs, args.split_rate, args.bs_ratio)
    
    train_op.train()
    train_op.feature_selection()
    file_name = args.classifier + args.file_name
    # train_op.save_results(file_name=file_name)

    

