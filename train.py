import torch

class Base_train(object):
    def __init__(self, model, x_data_0, x_data_1, y_data_0, y_data_1, epochs, ratio=0.8, bs_ratio=0.05):
        self.model = model

        self.x_data_0 = x_data_0
        self.x_data_1 = x_data_1
        self.y_data_0 = y_data_0
        self.y_data_1 = y_data_1

        self.epochs = epochs
        self.ratio = ratio
        self.bs_ratio = bs_ratio

    def step(self, x_train_0, x_train_1, x_test_0, x_test_1):
        train_loader_0 = torch.utils.data.DataLoader(x_train_0, batch_size=int(self.bs_ratio*x_train_0.shape[0]+1))
        train_loader_1 = torch.utils.data.DataLoader(x_train_1, batch_size=int(self.bs_ratio*x_train_1.shape[0]+1))

        for e in range(self.epochs):
            tr_loss_0 = 0
            tr_loss_1 = 0
            tr_loss = 0
            for (train_0, train_1) in zip(train_loader_0, train_loader_1):
                train_loss, train_con, total_loss = self.model.train(train_0, train_1)
                tr_loss_0 += train_con
                tr_loss_1 += train_loss
                tr_loss += total_loss
            if (e+1) % 20 == 0:
                print(f'Train {e+1} steps | 0 type loss: {tr_loss_0/len(train_loader_0)} | 1 type loss: {tr_loss_1/len(train_loader_0)} | total loss: {tr_loss/len(train_loader_0)} | lambda: {self.model.method.lamda}')
                test_loss, test_con = self.model.test(x_test_0, x_test_1)
                print(f'Test | 0 type loss: {test_con} | 1 type loss: {test_loss}')
        return train_loss, train_con, test_loss, test_con

    # Base train, Cross-validation, ...
    def train(self):
        tr_size = int(self.ratio * len(self.x_data_0))
        x_train_0, x_test_0 = self.x_data_0[0:tr_size, :], self.x_data_0[tr_size:-1, :]

        tr_size = int(self.ratio * len(self.x_data_1))
        x_train_1, x_test_1 = self.x_data_1[0:tr_size, :], self.x_data_1[tr_size:-1, :]

        self.step(x_train_0, x_train_1, x_test_0, x_test_1)

    # Feature selection
    def feature_selection(self):
        pass

    # Save results
    def save_results(self,filename='Default'):
        self.model.save(filename)