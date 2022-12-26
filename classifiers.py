import torch
import torch.nn as nn
import torch.nn.functional as F


class exp_linear(nn.Module):
    def __init__(self, dims,divide_dims=False):
        super(exp_linear, self).__init__()
        self.dims=1
        if(divide_dims):
            self.dims=dims
        self.linear = nn.Linear(dims, 1)

    def forward(self, x, inds=1.0):
        y = torch.log(1 + torch.exp( - inds * self.linear(x/self.dims)))
        return y
    
    def predict(self,x,probs=False):
        x=x/self.dims
        if(probs):
            y_1 = torch.log(1 + torch.exp( - self.linear(x))).detach()
            y_neg1=torch.log(1 + torch.exp( self.linear(x))).detach()
            # print(y_1)
            tmpf=lambda x:x[0]/(x[0]+x[1])
            return [tmpf((y_1[i][0],y_neg1[i][0])) for i in range(len(y_1))]
            
        y=self.linear(x)
        tmpf=lambda x:-1 if x[0]>=0 else 1
        return torch.tensor([tmpf(x) for x in y])

class logistic(nn.Module):
    def __init__(self, dims,divide_dims=False):
        super(logistic, self).__init__()
        self.dims=1
        if(divide_dims):
            self.dims=dims
        self.linear = nn.Linear(dims, 1)

    def forward(self, x, inds=1.0):
        x=x/self.dims
        y = 1/(1 + torch.exp(self.linear(x)))
        # print(y)
        if(inds<0):
            y=1-y
        return y
    
    def predict(self,x,probs=False):
        x=x/self.dims
        if(probs):
            y_1 = 1/(1 + torch.exp(self.linear(x))).detach()
            # print(y_1)
            tmpf=lambda x:x[0]
            return [tmpf(x) for x in y_1]
        y=self.linear(x)
        tmpf=lambda x:-1 if x[0]>=0 else 1
        return torch.tensor([tmpf(x) for x in y])
    
class DNN(nn.Module):
    def __init__(self, dims,divide_dims=False):
        super(DNN, self).__init__()
        self.dims=1
        if(divide_dims):
            self.dims=dims
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, inds=1.0):
        #y[0]代表预测为batch_0，y[1]代表预测为batch_1
        x=x/self.dims
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        # print(y[:,1])
        # print(y)
        if(inds<0):
            return y[:,1]
        return y[:,0]
    
    def predict(self,x,probs=False):
        x=x/self.dims
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        # print(y)
        if(probs):
            # print(y_1)
            return y[...,0]
        y_1=y[...,0]-y[...,1]
        tmpf=lambda x:1 if x>=0 else -1
        return [tmpf(x) for x in y_1]

class LDA(nn.Module):
    def __init__(self, dims,divide_dims=False):
        super(LDA, self).__init__()
        self.dims=1
        if(divide_dims):
            self.dims=dims
        self.linear1 = nn.Linear(dims, 1)
        self.linear2 = nn.Linear(dims, 1)

    def forward(self, x, inds=1.0):
        x=x/self.dims
        y = 1/(1 + torch.exp(self.linear1(x)+self.linear2(x**2)))
        # print(y)
        if(inds<0):
            y=1-y
        return y
    
    def predict(self,x,probs=False):
        x=x/self.dims
        if(probs):
            y_1 = 1/(1 + torch.exp(self.linear1(x)+self.linear2(x**2))).detach()
            # print(y_1)
            tmpf=lambda x:x[0]
            return [tmpf(x) for x in y_1]
        y=self.linear1(x)+self.linear2(x**2)
        tmpf=lambda x:-1 if x[0]>=0 else 1
        return torch.tensor([tmpf(x) for x in y])

