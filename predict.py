# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 22:26:32 2022

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
        ind_0 = np.reshape(data['y']==-1, -1)
        ind_1 = np.reshape(data['y']==1, -1)
    
    return x_data[ind_0, :], x_data[ind_1, :], y_data[ind_0, :], y_data[ind_1, :]



if __name__ == "__main__":

    args = config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_data_0, x_data_1, y_data_0, y_data_1 = load_data(args)
    model = Model(x_data_0.shape[1], args.delta, args.lr, 'logistic', args.method)
    
    model.load('logistic_003')
    predict_y0=model.predict(torch.tensor(x_data_0))
    predict_y1=model.predict(torch.tensor(x_data_1))
    errn_01=sum([predict_y0[i]!=y_data_0[i][0] for i in range(len(y_data_0))])
    print("正常邮件：","分类错误",errn_01,"分类正确",len(y_data_0)-errn_01)
    errn_10=sum([predict_y1[i]!=y_data_1[i][0] for i in range(len(y_data_1))])
    print("垃圾邮件：","分类错误",errn_10,"分类正确",len(y_data_1)-errn_10)
    
    predict_y0_probs=model.predict(torch.tensor(x_data_0[:10]),probs=True)#返回预测为1类的概率
    print(predict_y0_probs)
    
    
    model1 = Model(x_data_0.shape[1], args.delta, args.lr, 'linear', args.method)
    
    model1.load('exp_010')
    predict_y0=model1.predict(torch.tensor(x_data_0))
    predict_y1=model1.predict(torch.tensor(x_data_1))
    errn_01=sum([predict_y0[i]!=y_data_0[i][0] for i in range(len(y_data_0))])
    print("正常邮件：","分类错误",errn_01,"分类正确",len(y_data_0)-errn_01)
    errn_10=sum([predict_y1[i]!=y_data_1[i][0] for i in range(len(y_data_1))])
    print("垃圾邮件：","分类错误",errn_10,"分类正确",len(y_data_1)-errn_10)
    
    predict_y1_probs=model1.predict(torch.tensor(x_data_1[:10]),probs=True)#返回预测为1类的概率
    print(predict_y1_probs)

    


