import torch
import torch.nn as nn
import torch.nn.functional as F


class exp_linear(nn.Module):
    def __init__(self, dims):
        super(exp_linear, self).__init__()

        self.linear = nn.Linear(dims, 1)

    def forward(self, x, inds=1.0):
        y = torch.log(1 + torch.exp( - inds * self.linear(x)))
        return y
    
    def predict(self,x,probs=False):
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
    def __init__(self, dims):
        super(logistic, self).__init__()

        self.linear = nn.Linear(dims, 1)

    def forward(self, x, inds=1.0):
        y = 1/(1 + torch.exp(self.linear(x)))
        if(inds<0):
            y=1-y
        return y
    
    def predict(self,x,probs=False):
        if(probs):
            y_1 = 1/(1 + torch.exp(self.linear(x))).detach()
            # print(y_1)
            tmpf=lambda x:x[0]
            return [tmpf(x) for x in y_1]
        y=self.linear(x)
        tmpf=lambda x:-1 if x[0]>=0 else 1
        return torch.tensor([tmpf(x) for x in y])