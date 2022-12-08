import torch
import numpy as np
import classifiers as cf

class Model(object):
    def __init__(self, dims, delta, lr=1e-3, classifier='linear', method='lagrange'):
        if classifier == 'linear':
            self.classifier = cf.exp_linear(dims)
        if classifier == 'linear_logistic':
            self.classifier = cf.logistic(dims)
        
        if method == 'lagrange':
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

        return loss_0.cpu().detach().item(), loss_1.cpu().detach().item(), npc_loss.cpu().detach().item()

    def test(self, x_batch_0, x_batch_1, y_batch_0=None, y_batch_1=None):
        with torch.no_grad():
            test_loss, test_con = self.classifier(torch.tensor(x_batch_0), -1.0).cpu().mean(), self.classifier(torch.tensor(x_batch_1), 1.0).cpu().mean()
        return test_loss.item(), test_con.item()

    def save(self, filename):
        torch.save(self.classifier.state_dict(), filename + "_params")

    def load(self, filename):
        self.classifier.load_state_dict(torch.load(filename + "_params"))
    def predict(self,x):
        return self.classifier.predict(x)



class Lagrange(object):
    def __init__(self, delta, rho=1e-2, initial=1.0):
        self.delta = delta
        self.lamda = initial
        self.rho = rho
    
    def weight_loss(self, loss_0, loss_1):
        # return self.lamda * (loss_0 - self.delta) + loss_1
        return self.lamda * loss_0 + loss_1

    def update(self, loss_0, loss_1):
        dir = loss_0.cpu().detach().item() - self.delta
        self.lamda = max(0, self.lamda + self.rho * dir)

        
