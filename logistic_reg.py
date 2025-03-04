import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import random
from math import sqrt

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# 构造与处理数据集
def prepare_process(data):
    """读取特征并处理数据"""
    features = ['track_popularity', 'danceability', 'energy', 'key', 'loudness',\
            'mode', 'speechiness', 'acousticness', 'instrumentalness',\
            'liveness', 'valence', 'tempo', 'duration_ms']
    label = ['playlist_genre']           
    features_df = data.loc[:, features]
    label_df = data.loc[:, label]
    mapdict = {'pop': [1, 0, 0, 0, 0 ,0], 
               'rap': [0, 1, 0, 0, 0 ,0], 
               'edm': [0, 0, 1, 0, 0 ,0], 
               'rock':[0, 0, 0, 1, 0 ,0], 
               'latin':[0, 0, 0, 0, 1 ,0], 
               'r&b': [0, 0, 0, 0, 0 ,1] }
    label_value = label_df.values.reshape(len(label_df))
    label_value = np.array([mapdict[x] for x in label_value])
    features_value = features_df.values
    features_value = (features_value - features_value.min(axis=0, keepdims=True)) /\
        (features_value.max(axis=0, keepdims=True) - \
         features_value.min(axis=0, keepdims=True))
    return features_value, label_value 
    

class SpotifyDataset(Dataset):
    def __init__(self, datapath='spotify_songs.csv'):
        self.data = pd.read_csv(datapath)
        self.data = self.data.dropna()
        self.features, self.label = prepare_process(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx, :], self.label[idx]

    def __len__(self):
        return len(self.data)

# 构建模型
class VanillaLogReg():
    def  __init__(self, init_theta):
        self.theta = init_theta

    def predict(self, X):
        """
        预测函数，根据输入的特征矩阵 X 和训练好的 theta 进行预测
        :param X: 输入特征矩阵
        :param theta: 模型参数
        :return: 预测结果
        """
        z = torch.mm(X, self.theta)
        pred_logit = softmax(z)
        return pred_logit

    def compute_gradient(self, X, pred, label):
        """
        计算梯度
        :prarm X: 输入特征矩阵
        :param pred: 预测结果
        :prarm label: 真实标签
        :return: 计算得到的梯度
        """
        return torch.mm(X.T, (pred - label))

    def loss(self, pred, label):
        """
        计算对数似然损失函数
        :param pred: 预测值
        :param label: 真实值
        :return: 计算得到的损失值
        """
        return (-label * torch.log(pred)).sum()
    
    def __str__(self):
        return 'VanillaLogReg'

    def __call__(self, feature, label, theta=None):
        if theta != None:
            self.theta = theta
        pred = self.predict(feature)
        gradient = self.compute_gradient(feature, pred, label)
        return pred, gradient

class VanillaLogRegNorm(VanillaLogReg):
    def __init__(self, init_theta, norm_weight):
        super().__init__(init_theta)
        self.norm_weight = norm_weight

    def compute_gradient(self, X, pred, label):
        return super().compute_gradient(X, pred, label) + \
        self.norm_weight * 2 * self.theta
    
    def loss(self, pred, label):
        return super().loss(pred, label) + \
            (self.norm_weight * self.theta * self.theta).sum()

    def __str__(self):
        return 'VanillaLogRegNorm'

    def __call__(self, feature, label, theta=None):
        if theta != None:
            self.theta = theta
        return super().__call__(feature, label)

def softmax(z):
    """
    Sigmoid 函数，将输入的 z 值映射到 [0, 1] 之间
    :param z: 输入值
    :return: 经过 softmax 函数处理后的结果
    """
    return torch.exp(z) / torch.sum(torch.exp(z), axis=-1, keepdims=True)

def accuracy(pred, label):
    """
    计算准确率
    :param pred: 预测结果
    :param label: 真实标签
    """
    acc = torch.all(label==pred, axis=-1).sum().item() / len(label)
    return acc

def get_label_from_pred(pred):
    """从预测结果中获取真实标签"""
    return (pred == torch.max(pred, axis=-1, keepdim=True).values).to(int)

def train(train_dataloader, opt, model, epoch, getacc=False, ifprt=False):
    """
    梯度下降算法，用于更新参数 theta
    :param train_dataloader: 训练集 
    :param theta: 模型参数初始值
    :param lr: 学习率
    :param epoch: 训练迭代次数
    :param opt: 优化器
    :param model: 模型
    :return: 更新后的theta和每次迭代的损失列表
    """
    loss_history = []
    if getacc:
        acc_history = []
    for _ in range(epoch):
        for batch_feature, batch_label in train_dataloader:
            batch_size = len(batch_feature)
            if isinstance(opt, Nesterov_SGD):
                theta, pred = opt.step(model, batch_feature, batch_label, 
                                       batch_size)
            else:
                pred, gradient = model(batch_feature, batch_label)
                gradient = gradient / batch_size
                model.theta = opt.step(gradient)
            loss_history.append(model.loss(pred, batch_label) / batch_size)
            if getacc:
                acc_history.append(accuracy(get_label_from_pred(pred), 
                                            batch_label))
        if ifprt == True:
            epoch_acc = accuracy(get_label_from_pred(pred), batch_label)
            print(f'完成第{_ + 1}/{epoch}个Epoch,当前准确率{epoch_acc * 100:.2f}%')
    if getacc:
        return model, loss_history, acc_history
    else:
        return model, loss_history

def eval(valid_dataloader, model):
    """
    评估训练结果
    :param valid_dataloader: 验证集
    :param model: 训练结果
    """
    loss_history = []
    accuracy_history = []
    for batch_feature, batch_label in valid_dataloader:
        batch_size = len(batch_feature)
        pred, _ = model(batch_feature, batch_label)
        loss_history.append(model.loss(pred, batch_label) / batch_size)
        accuracy_history.append(accuracy(get_label_from_pred(pred), batch_label))
    return np.array(accuracy_history), np.array(loss_history)

# 定义优化器
class Vanilla_SGD():
    def __init__(self, theta, lr):
        self.theta = theta
        self.lr = lr
    
    def step(self, gradient):
        self.theta -= self.lr * gradient
        return self.theta
    
    def __str__(self):
        return 'SGD'

class Momentum_SGD():
    def __init__(self, theta, lr):
        self.theta = theta
        self.lr = lr
        self.m = 0
        self.beta = 0.9  # 默认设置为0.9

    def step(self, gradient):
        if isinstance(self.m, int):
            self.m = gradient
            self.theta -= self.lr * self.m
            return self.theta
        else:
            self.m = self.beta * self.m + (1 - self.beta) * gradient
            self.theta -= self.lr * self.m
        return self.theta
    
    def __str__(self):
        return 'Momentum'

class Nesterov_SGD():
    def __init__(self, theta, lr):
        self.theta = theta
        self.lr = lr
        self.miu = 0
        self.v = self.theta
        self.k = 1

    def step(self, model, feature, label, size):
        if self.k == 1:
            pred, gradient = model(feature, label, self.theta)
            gradient = gradient / size
            self.theta = self.theta - self.lr * gradient 
            self.v -= self.theta
            self.k += 1
            self.miu = (self.k - 1) / (self.k + 2)
            return self.theta, pred  
        else:
            pred, gradient = model(feature, label, self.theta + 
                                   self.miu * self.v)
            gradient = gradient / size
            self.v = self.miu * self.v - self.lr * gradient 
            self.theta += self.v
            self.k += 1
            self.miu = (self.k - 1) / (self.k + 2)
            return self.theta, pred

    def __str__(self):
        return 'Nesterov'

class AdaGrad_SGD():
    def __init__(self, theta, lr):
        self.theta = theta
        self.lr = lr
        self.eps = 1e-6
        self.g = 0

    def step(self, gradient):
        self.g += gradient * gradient
        self.theta -= self.lr * gradient / torch.sqrt(self.g + self.eps)
        return self.theta
        
    def __str__(self):
        return 'AdaGrad'

class RMSProp_SGD():
    def __init__(self, theta, lr):
        self.theta = theta
        self.lr = lr
        self.m = 0
        self.eps = 1e-6
        self.rho = 0.99

    def step(self, gradient):
        if isinstance(self.m, int):
            self.theta -= self.lr * gradient
            self.m = gradient * gradient
            return self.theta
        else:
            self.theta -= self.lr * gradient / torch.sqrt(self.m + self.eps)
            self.m = self.rho * self.m + (1 - self.rho) * gradient * gradient
            return self.theta

    def __str__(self):
        return 'RMSProp'
    

class AdaDelta_SGD():
    def __init__(self, theta, lr):
        self.theta = theta
        self.lr = lr
        self.deltax = 0
        self.d = 0
        self.m = 0
        self.eps = 1e-6
        self.rho = 0.9

    def step(self, gradient):
        if isinstance(self.m ,int):
            self.m = gradient * gradient
            self.deltax = - self.lr * gradient / torch.sqrt(self.m + self.eps)
            self.theta += self.deltax
            self.d = self.deltax * self.deltax
            return self.theta
        else:
            self.m = self.rho * self.m + (1 - self.rho) * gradient * gradient
            self.deltax = - torch.sqrt(self.d + self.eps) / torch.sqrt(self.m +
                            self.eps) * gradient
            self.theta += self.deltax
            self.d = self.rho * self.d + (1 - self.rho) * self.deltax * \
                self.deltax
            return self.theta

    def __str__(self):
        return 'AdaDelta'

class Adam_SGD():
    def __init__(self, theta, lr):
       self.theta = theta
       self.lr = lr
       self.beta1 = 0.8
       self.beta2 = 0.9
       self.eps = 1e-6
       self.m = 0
       self.v = 0
       self.k = 0

    def step(self, gradient):
        if self.k == 0:
            self.m = gradient
            self.v = gradient * gradient
            self.theta -= self.lr * self.m / torch.sqrt(self.v + self.eps)
            self.k += 1
            return self.theta
        else:
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * gradient *gradient
            m = self.m / (1 - self.beta1 ** self.k)
            v = self.v / (1 - self.beta2 ** self.k)
            self.theta -= self.lr * m / torch.sqrt(v + self.eps)
            self.k += 1
            return self.theta

    def __str__(self):
        return 'Adam'

class AdaFactor_SGD():
    def __init__(self, theta, lr):
        self.theta = theta
        self.eps1 = 1e-30
        self.eps2 = lr
        self.d = 1
        self.rho = lambda t: min(1e-2, 1 / sqrt(t))
        self.beta2 = lambda t: 1 - t ** (-0.8 * t)
        self.t = 1
    
    def RMS(self, mat):
        return mat.norm(2) / mat.numel() ** 0.5


    def step(self, gradient):
        if self.t == 1:
            self.irow = torch.ones(gradient.shape[0], dtype=torch.float64)
            self.icol = torch.ones(gradient.shape[1], dtype=torch.float64)
            alpha = max(self.eps2, self.RMS(self.theta)) * self.rho(self.t)
            self.r = (gradient * gradient + self.eps1) @ self.icol
            self.c = self.irow @ (gradient * gradient + self.eps1)
            self.v = torch.outer(self.r, self.c) / (self.irow @ self.r)
            self.u = gradient / torch.sqrt(self.v)
            self.u = self.u / max(1, self.RMS(self.u) / self.d)
            self.theta = - alpha * self.u
            self.t += 1
        else:
            alpha = max(self.eps2, self.RMS(self.theta)) * self.rho(self.t)
            self.r = self.beta2(self.t) * self.r + (1 - self.beta2(self.t)) * \
            (gradient * gradient + self.eps1) @ self.icol
            self.c = self.beta2(self.t) * self.c + (1 - self.beta2(self.t)) * \
            self.irow @ (gradient * gradient + self.eps1) 
            self.v = torch.outer(self.r, self.c) / (self.irow @ self.r)
            self.u = gradient / torch.sqrt(self.v)
            self.u = self.u / max(1, self.RMS(self.u) / self.d)
            self.theta -= alpha * self.u
            self.t += 1
        return self.theta
    
    def __str__(self):
        return 'Adafactor'

def parse_args(args):
    parser = argparse.ArgumentParser(description='Using logistic regression to\
                                     calssify music in spotify into 6 classes.')
    parser.add_argument('--output', type=str, default='output.txt')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bthsz', type=int, default=1024, help='batch size')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--datapath', type=str, default='spotify_songs.csv')
    parser.add_argument('--optimizer', type=str, default='AdaFactor_SGD', 
                        choices=['Vanilla_SGD', 'Momentum_SGD', 'Nesterov_SGD', \
                                 'AdaGrad_SGD', 'RMSProp_SGD', 'AdaDelta_SGD', \
                                 'Adam_SGD', 'AdaFactor_SGD'])
    parser.add_argument('--model', type=str, default='VanillaLogReg')
    parser.add_argument('--norm_weight', type=float, default=None)
    args = parser.parse_args(args)
    return args


# 数据可以从下面地址下载 
# https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs

def main(args):
    """
    建模，训练，评估 
    我们提供一个demo来验证模型有效性, 覆写main函数来验证
    def main():
        X = torch.from_numpy(np.random.randn(10000, 3))
        theta_true = torch.from_numpy(np.array([[-1, 2, -3], [2, -1, 3], [3, 1, -2]])).to(torch.float64)
        probabilities = predict(X, theta_true)
        y = get_label_from_pred(probabilities)
        theta = torch.rand(X.shape[1], 3, dtype=torch.float64)
        alpha = 0.01
        class testdataset(DataLoader):
            def __init__(self, X, y):
                self.data = X
                self.label = y
            def __getitem__(self, idx):
                return self.data[idx, :], self.label[idx, :]
            def __len__(self):
                return len(self.data) 
        X_test = torch.from_numpy(np.random.randn(100, 3)).to(torch.float64)
        y_test = predict(X_test, theta_true)
        y_test = get_label_from_pred(y_test)
        testtrain, testvalid = testdataset(X, y), testdataset(X_test, y_test)
        testdataloader = DataLoader(testtrain, batch_size=1024, shuffle=True)
        validataloader = DataLoader(testvalid, batch_size=1024, shuffle=True)
        theta = train(testdataloader, theta, 0.01, vanilla_GD, 100, True)
        acc_list, loss_list = eval(validataloader, theta)
        print(f'测试集平均正确率{np.mean(acc_list)*100:.2f}%')
        print(f'平均损失{np.mean(loss_list):.2f}')
    正确率应该超过95%
    """
    # Setting 
    print('优化器:', args.optimizer)
    print('BatchSize:', args.bthsz)
    print('Epoch:', args.epoch)
    print('learning rate:', args.lr)
    print('model:', args.model)
    spotify_data = SpotifyDataset(args.datapath)
    print('完成数据加载')
    trainset, validset = random_split(spotify_data, lengths=[0.9, 0.1])
    print('完成数据分割')
    train_dataloader = DataLoader(trainset, batch_size=args.bthsz, 
                                  shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=args.bthsz, 
                                  shuffle=True)
    theta_size = trainset[0][0].shape[0]
    theta_init = torch.rand(theta_size, 6, dtype=torch.float64)
    optmap = {'Vanilla_SGD': Vanilla_SGD,
              'Momentum_SGD': Momentum_SGD,
              'Nesterov_SGD': Nesterov_SGD,
              'AdaGrad_SGD': AdaGrad_SGD,
              'RMSProp_SGD':RMSProp_SGD,
              'AdaDelta_SGD': AdaDelta_SGD,
              'Adam_SGD': Adam_SGD,
              'AdaFactor_SGD': AdaFactor_SGD}
    modelmap = {'VanillaLogReg': VanillaLogReg,
                'VanillaLogRegNorm': VanillaLogRegNorm}
    model = modelmap[args.model]
    if args.norm_weight != None:
        model = model(theta_init, args.norm_weight)
    else:
        model = model(theta_init)
    optimizer = optmap[args.optimizer]
    opt = optimizer(theta_init, args.lr)

    # Train
    model, _ = train(train_dataloader, 
                             opt,
                             model,  
                             args.epoch,
                             ifprt=True
                             )
    print(f'完成训练, 开始测试')

    # Eval
    acc_list, loss_list = eval(valid_dataloader, model)
    print(f'测试集平均正确率{np.mean(acc_list)*100:.2f}%')
    print(f'平均损失{np.mean(loss_list):.2f}')


if __name__ == '__main__':
    print('Starting Script')
    args = parse_args(None)
    main(args)


