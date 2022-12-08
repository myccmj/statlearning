import scipy.io as sio
import numpy as np
import torch
import argparse

from algs import Model
import train

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="spambase", choices=['spambase', 'gisette', 'madelon'])
    parser.add_argument("--delta", default=0.15, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--classifier", default="linear")                 
    parser.add_argument("--method", default="lagrange")
    parser.add_argument("--train_type", default="base")
    parser.add_argument("--ratio", default=0.8, type=float)
    parser.add_argument("--bs_ratio", default=0.05, type=float)
    parser.add_argument("--seed", default=0, type=int)             
    parser.add_argument("--epochs", default=1000, type=int)
    args = parser.parse_args()

    return args



def load_data(args):
    if args.dataset == 'spambase':
        data = sio.loadmat('./data/spambase.mat')
        # print(data)
        x_data = data['X'].astype(np.float32)
        y_data = data['y'].astype(np.float32)
        # print(data['y']==-1)
        ind_0 = np.reshape(data['y']==-1, -1)
        # print(ind_0)
        ind_1 = np.reshape(data['y']==1, -1)
    
    return x_data[ind_0, :], x_data[ind_1, :], y_data[ind_0, :], y_data[ind_1, :]



if __name__ == "__main__":

    args = config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_data_0, x_data_1, y_data_0, y_data_1 = load_data(args)
    print(x_data_0, x_data_1, y_data_0, y_data_1)
    # model = Model(x_data_0.shape[1], args.delta, args.lr, 'linear', args.method)
    model = Model(x_data_0.shape[1], args.delta, args.lr, 'linear_logistic', args.method)
    if args.train_type == 'base':
        train_op = train.Base_train(model, x_data_0, x_data_1, y_data_0, y_data_1, args.epochs, 1, args.bs_ratio)
    train_op.train()
    train_op.feature_selection()
    # train_op.save_results('logistic')
    
    #predict
    # model_logistic.load('logistic')
    # predict_y0=model_logistic.predict(torch.tensor(x_data_0))
    # predict_y1=model_logistic.predict(torch.tensor(x_data_1))
    # errn_01=sum([predict_y0[i]!=y_data_0[i] for i in range(len(y_data_0))])
    # print(errn_01,len(y_data_0)-errn_01)
    # errn_10=sum([predict_y1[i]!=y_data_1[i] for i in range(len(y_data_1))])
    # print(errn_10,len(y_data_1)-errn_10)
    
    # model_exp.load('exp_linear')
    # predict_y0=model_exp.predict(torch.tensor(x_data_0))
    # predict_y1=model_exp.predict(torch.tensor(x_data_1))
    # errn_01=sum([predict_y0[i]!=y_data_0[i] for i in range(len(y_data_0))])
    # print(errn_01,len(y_data_0)-errn_01)
    # errn_10=sum([predict_y1[i]!=y_data_1[i] for i in range(len(y_data_1))])
    # print(errn_10,len(y_data_1)-errn_10)
    

    

