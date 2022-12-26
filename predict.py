# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:10:00 2022

@author: myccm
"""

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
    model = Model(x_data_0.shape[1], 0.12, args.lr, 'linear_logistic', args.method)
    
    model_param=['exp_linear','logistic_009','logistic','logistic_015','logistic_018','logistic_010']
    for param in model_param:
        model.load(param)
        predict_y1=model.predict(torch.tensor(x_data_1[:10]),1)
        print(predict_y1)
        predict_y0=model.predict(torch.tensor(x_data_0))
        predict_y1=model.predict(torch.tensor(x_data_1))
        errn_01=sum([predict_y0[i]!=y_data_0[i][0] for i in range(len(y_data_0))])
        print("垃圾邮件：","分类错误",errn_01,"分类正确",len(y_data_0)-errn_01)
        errn_10=sum([predict_y1[i]!=y_data_1[i][0] for i in range(len(y_data_1))])
        print("正常邮件：","分类错误",errn_10,"分类正确",len(y_data_1)-errn_10)
        print('//////////////////////////////')
    model = Model(x_data_0.shape[1], 0.12, args.lr, 'LDA', args.method)
    model.load('LDA_010')
    predict_y1=model.predict(torch.tensor(x_data_1[:10]),1)
    print(predict_y1)
    predict_y0=model.predict(torch.tensor(x_data_0))
    predict_y1=model.predict(torch.tensor(x_data_1))
    print(predict_y0[:10])
    print(predict_y1[:10])
    errn_01=sum([predict_y0[i]!=y_data_0[i][0] for i in range(len(y_data_0))])
    print("垃圾邮件：","分类错误",errn_01,"分类正确",len(y_data_0)-errn_01)
    errn_10=sum([predict_y1[i]!=y_data_1[i][0] for i in range(len(y_data_1))])
    print("正常邮件：","分类错误",errn_10,"分类正确",len(y_data_1)-errn_10)
    print('//////////////////////////////')
    model = Model(x_data_0.shape[1], 0.12, args.lr, 'DNN', args.method)
    model.load('DNN_006')
    predict_y1=model.predict(torch.tensor(x_data_1[:10]),1)
    print(predict_y1)
    predict_y0=model.predict(torch.tensor(x_data_0))
    predict_y1=model.predict(torch.tensor(x_data_1))
    print(predict_y0[:10])
    print(predict_y1[:10])
    errn_01=sum([predict_y0[i]!=y_data_0[i][0] for i in range(len(y_data_0))])
    print("垃圾邮件：","分类错误",errn_01,"分类正确",len(y_data_0)-errn_01)
    errn_10=sum([predict_y1[i]!=y_data_1[i][0] for i in range(len(y_data_1))])
    print("正常邮件：","分类错误",errn_10,"分类正确",len(y_data_1)-errn_10)
    print('//////////////////////////////')

    

