import torch
import numpy as np
import classifiers as cf

class Model(object):
    def __init__(self, dims, delta, lr=1e-3, classifier='linear', method='lagrange',divide_dims=False):
        if classifier == 'linear':
            self.classifier = cf.exp_linear(dims,divide_dims)
        if classifier == 'linear_logistic':
            self.classifier = cf.logistic(dims,divide_dims)
        if classifier == 'DNN':
            self.classifier = cf.DNN(dims,divide_dims)
        if classifier == 'LDA':
            self.classifier = cf.LDA(dims,divide_dims)
        
        if method == 'base':
            self.method = Base()
        elif method == 'lagrange':
            self.method = Lagrange(delta)
        
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.delta = delta

    def train(self, x_batch_0, x_batch_1, y_batch_0=None, y_batch_1=None):
        self.optimizer.zero_grad()
        loss_0, loss_1 = self.classifier(x_batch_0, -1.0).mean(), self.classifier(x_batch_1, 1.0).mean()
        npc_loss = self.method.weight_loss(loss_0, loss_1)
        npc_loss.backward()
        self.optimizer.step()

        self.method.update(loss_0, loss_1)

        return loss_0.detach().item(), loss_1.detach().item(), (loss_0+loss_1).detach().item()

    # def test(self, x_batch_0, x_batch_1, y_batch_0=None, y_batch_1=None):
    #     with torch.no_grad():
    #         loss_0, loss_1 = self.classifier(torch.tensor(x_batch_0), -1.0).mean(), self.classifier(torch.tensor(x_batch_1), 1.0).mean()
    #     return loss_0.item(), loss_1.item()
    def test(self, x_batch_0, x_batch_1, y_batch_0=None, y_batch_1=None):
        with torch.no_grad():
            loss_0_0, loss_0_1 = self.classifier(torch.tensor(x_batch_0), -1.0), self.classifier(torch.tensor(x_batch_0), 1.0)
            loss_1_0, loss_1_1 = self.classifier(torch.tensor(x_batch_1), -1.0), self.classifier(torch.tensor(x_batch_1), 1.0)
            loss_0 = len(loss_0_0[loss_0_0.gt(loss_0_1)])/max(len(loss_0_0),1)
            loss_1 = len(loss_1_1[loss_1_1.gt(loss_1_0)])/max(len(loss_1_1),1)
        return loss_0, loss_1, loss_0 + loss_1

    def save(self, filename):
        torch.save(self.classifier.state_dict(), filename + "_params")

    def load(self, filename):
        self.classifier.load_state_dict(torch.load(filename + "_params"))
    def predict(self,x,probs=False):
        return self.classifier.predict(x,probs)



class Base(object):
    def __init__(self):
        pass
    
    def weight_loss(self, loss_0, loss_1):
        return loss_0 + loss_1

    def update(self, loss_0, loss_1):
        pass


class Lagrange(object):
    def __init__(self, delta, rho=1e-2, initial=1.0):
        self.delta = delta
        self.lamda = initial
        self.rho = rho
    
    def weight_loss(self, loss_0, loss_1):
        return self.lamda *loss_0 + loss_1

    def update(self, loss_0, loss_1):
        dir = loss_0.detach().item() - self.delta
        self.lamda = max(0.1, self.lamda + self.rho * dir)

        
