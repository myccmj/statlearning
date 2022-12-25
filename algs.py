import torch
import torch.nn.functional as F
import numpy as np
import classifiers as cf

class Model(object):
    def __init__(self, dims, delta, lr=1e-3, classifier='linear', method='lagrange'):
        if classifier == 'linear':
            self.classifier = cf.exp_linear(dims)
        if classifier == 'logistic':
            self.classifier = cf.logistic(dims)
        
        if method == 'base':
            self.method = Base()
        elif method == 'lagrange':
            self.method = Lagrange(delta)
        elif method == 'csa':
            self.method = CSA(delta)
        elif method == 'penalty':
            self.method = Penalty(delta)
        elif method == 'penalty2':
            self.method = PenaltyV2(delta)
        
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
            loss_0 = torch.abs(self.classifier.predict(torch.tensor(x_batch_0))-1).mean()/2.0
            loss_1 = torch.abs(self.classifier.predict(torch.tensor(x_batch_1))+1).mean()/2.0
        return loss_0, loss_1, loss_0 + loss_1

    def save(self, filename):
        torch.save(self.classifier.state_dict(), filename + "_params")

    def load(self, filename):
        self.classifier.load_state_dict(torch.load(filename + "_params"))



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
        return self.lamda * (loss_0 - self.delta) + loss_1

    def update(self, loss_0, loss_1):
        dir = loss_0.detach().item() - self.delta
        self.lamda = max(0, self.lamda + self.rho * dir)


class CSA(object):
    def __init__(self, delta, initial=1.25):
        self.delta = initial * delta
    
    def weight_loss(self, loss_0, loss_1):
        if loss_0 > self.delta:
            return loss_0
        else:
            return loss_1

    def update(self, loss_0, loss_1):
        pass


class Penalty(object):
    def __init__(self, delta):
        self.weight = 1.0/(delta+1e-6)
        self.delta = delta
    
    def weight_loss(self, loss_0, loss_1):
        return self.weight * F.relu(loss_0-self.delta) + loss_1

    def update(self, loss_0, loss_1):
        pass


class PenaltyV2(object):
    def __init__(self, delta, rho=1.1, initial=1.5):
        self.delta = delta
        self.delta_pen = initial * delta
        self.rho = rho
        self.weight = 1.0
        self.total_it = 0
    
    def weight_loss(self, loss_0, loss_1):
         return self.weight * F.relu(loss_0-self.delta) + loss_1

    def update(self, loss_0, loss_1):
        self.total_it += 1
        if loss_0.detach().item() > self.delta_pen and self.total_it % 50 == 0:
            self.weight = min(self.weight * self.rho, 1.0/(self.delta+1e-6))

        
