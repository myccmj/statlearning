import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# x_data = sio.loadmat('./data/gisette/gisette_train_scale.mat')
# y_data = sio.loadmat('./data/gisette/gisette_labels.mat')
# print(x_data['A'].shape)

model_name = 'spambase'
# model_name = 'gisette'

lag = sio.loadmat(f'./{model_name}linearlagrange.mat')
pv1 = sio.loadmat(f'./{model_name}linearpenalty.mat')
pv2 = sio.loadmat(f'./{model_name}linearpenalty2.mat')
csa = sio.loadmat(f'./{model_name}linearcsa.mat')

x = 100*(np.arange(10) + 1)

delta = 0.1
c = delta*np.ones_like(x)

index = ['train_loss_0', 'train_loss_1', 'test_loss_0', 'test_loss_1']
names = ['0 type loss (train)', '1 type loss (train)', '0 type loss (test)', '1 type loss (test)']

for ind, name in zip(index, names):
    plt.figure(figsize=(10,8),dpi=200)

    plt.xlabel('Epochs', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.title(f"{name}", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if name == '0 type loss (train)':
        # plt.plot(x, np.reshape(np.log(lag[ind]+1e-10), [-1]), label='Lagrange',marker='o')
        # plt.plot(x, np.reshape(np.log(csa[ind]+1e-10), [-1]), label='CSA',marker='o')
        # plt.plot(x, np.reshape(np.log(pv1[ind]+1e-10), [-1]), label='Penalty',marker='o')
        # plt.plot(x, np.reshape(np.log(pv2[ind]+1e-10), [-1]), label='PenaltyV2',marker='o')
        # plt.plot(x, np.log(c+1e-10), linestyle='--')
        plt.plot(x, np.reshape(lag[ind], [-1]), label='Lagrange',marker='o')
        plt.plot(x, np.reshape(csa[ind], [-1]), label='CSA',marker='o')
        plt.plot(x, np.reshape(pv1[ind], [-1]), label='Penalty',marker='o')
        plt.plot(x, np.reshape(pv2[ind], [-1]), label='PenaltyV2',marker='o')
        plt.plot(x, c, linestyle='--')
        plt.legend(loc='lower left', fontsize=20)
    if name == '1 type loss (train)':
        # plt.plot(x, np.reshape(np.log(lag[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(csa[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(pv1[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(pv2[ind]+1e-10), [-1]),marker='o')
        plt.plot(x, np.reshape(lag[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(csa[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(pv1[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(pv2[ind], [-1]),marker='o')
    if name == '0 type loss (test)':
        # plt.plot(x, np.reshape(np.log(lag[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(csa[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(pv1[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(pv2[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.log(c+1e-10), linestyle='--')
        plt.plot(x, np.reshape(lag[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(csa[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(pv1[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(pv2[ind], [-1]),marker='o')
        plt.plot(x, c, linestyle='--')
    if name == '1 type loss (test)':
        # plt.plot(x, np.reshape(np.log(lag[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(csa[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(pv1[ind]+1e-10), [-1]),marker='o')
        # plt.plot(x, np.reshape(np.log(pv2[ind]+1e-10), [-1]),marker='o')
        plt.plot(x, np.reshape(lag[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(csa[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(pv1[ind], [-1]),marker='o')
        plt.plot(x, np.reshape(pv2[ind], [-1]),marker='o')

    plt.savefig(f'{model_name}_{name}.jpg', bbox_inches='tight')
    plt.clf()
    


