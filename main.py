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
    parser.add_argument("--bs_ratio", default=0.05, type=float)
    parser.add_argument("--seed", default=0, type=int)             
    parser.add_argument("--epochs", default=1000, type=int)
    args = parser.parse_args()

    return args



def load_data(args):
    if args.dataset == 'spambase':
        data = sio.loadmat('./data/spambase.mat')
        x_data = data['X'].astype(np.float32)
        y_data = data['y'].astype(np.float32)
        ind_0 = np.reshape(data['y']==1, -1)#垃圾邮件
        ind_1 = np.reshape(data['y']==-1, -1)#正常邮件
    elif args.dataset == 'gisette':
        x_data = np.loadtxt('./data/gisette_train.data').astype(np.float32)
        y_data= np.loadtxt('./data/gisette_train.labels').astype(np.float32)
        ind_0 = np.reshape(y_data==1, -1)#垃圾邮件
        ind_1 = np.reshape(y_data==-1, -1)#正常邮件
        return x_data[ind_0, :], x_data[ind_1, :], y_data[ind_0].reshape(-1,1), y_data[ind_1].reshape(-1,1)
    
    return x_data[ind_0, :], x_data[ind_1, :], y_data[ind_0, :], y_data[ind_1, :]



if __name__ == "__main__":

    args = config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # args.dataset='gisette'
    x_data_0, x_data_1, y_data_0, y_data_1 = load_data(args)
    print(x_data_0.shape,y_data_0.shape,x_data_1.shape,y_data_1.shape)
    print(x_data_0[:5])
    model = Model(x_data_0.shape[1], 0.1, args.lr, 'LDA', args.method,divide_dims=1 if args.dataset=='gisette' else 0)
    if args.train_type == 'base':
        train_op = train.Base_train(model, x_data_0, x_data_1, y_data_0, y_data_1, 1000, 1, args.bs_ratio)
    
    train_loss_0, train_loss_1, test_loss_0, test_loss_1=train_op.train()
    train_loss_0=np.array(train_loss_0)
    train_loss_1=np.array(train_loss_1)
    test_loss_0=np.array(test_loss_0)
    test_loss_1=np.array(test_loss_1)
    # np.save('LDA_010_tr_loss0',train_loss_0)
    # np.save('LDA_010_tr_loss1',train_loss_1)
    # np.save('LDA_010_ts_loss0',test_loss_0)
    # np.save('LDA_010_ts_loss1',test_loss_1)
    # train_op.feature_selection()
    train_op.save_results('LDA_010')

    

