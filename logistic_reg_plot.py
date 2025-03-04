import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch 
from logistic_reg import train
from torch.utils.data import DataLoader
import copy
import os
import argparse
from logistic_reg import *
import matplotlib.font_manager as fm

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


font_path = '/usr/local/share/fonts/WindowsFonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)

# 设置全局字体属性
plt.rcParams['font.family'] = font_prop.get_name()

def compare_different_optimizer(model, optimizer_list, output_path, epoch,
                                TrainDataloader, ValidDataloader):
    """比较不同优化器的收敛速度"""
    output_path = os.path.join(output_path, 'compare_diff_opt.png')
    record_trainloss = {}
    record_trainacc = {}
    length_loss = 0
    length_acc = 0
    for opt in optimizer_list:
        opt = copy.deepcopy(opt)
        model_opt = copy.deepcopy(model) 
        _, loss_history, acc_history = train(TrainDataloader, opt, model_opt, 
                                             epoch, True)
        if length_loss == 0:
            length_loss = len(loss_history)
            length_acc = len(acc_history)
        record_trainloss[str(opt)] = loss_history
        record_trainacc[str(opt)] = acc_history
    x = np.array(range(1, length_loss+1))
    plt.figure(figsize=(12, 4))
    plt.subplot(121) 
    plt.xlabel('迭代步', fontproperties=font_prop)
    plt.ylabel('损失值', fontproperties=font_prop)
    plt.grid(True)
    for opt_name in record_trainloss.keys():
        #plt.plot(x, record_trainloss[opt_name], linewidth=2.0, label=opt_name,
        #       alpha=0.3)
        #color = plt.gca().lines[-1].get_color()
        ewm = pd.Series(record_trainloss[opt_name]).ewm(alpha=0.2).mean()
        plt.plot(x, ewm, label=opt_name)
        color = plt.gca().lines[-1].get_color()
        plt.axhline(y=ewm.iat[-1], color=color, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left') 
    #plt.title('Loss in Trainset')
    plt.subplot(122)
    x = np.array(range(1, length_acc+1), dtype=int)
    plt.xlabel('迭代步', fontproperties=font_prop)
    plt.ylabel('准确率', fontproperties=font_prop)
    plt.yscale('linear')
    plt.grid(True)
    for opt_name in record_trainacc.keys():
        #plt.plot(x, record_trainacc[opt_name], linewidth=2.0, label=opt_name,
                 #alpha=0.3)
        #color = plt.gca().lines[-1].get_color()
        ewm = pd.Series(record_trainacc[opt_name]).ewm(alpha=0.2).mean()
        plt.plot(x, ewm, label=opt_name)
        color = plt.gca().lines[-1].get_color()
        plt.axhline(y=ewm.iat[-1], color=color, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')
    #plt.title('Loss in Validset')
    plt.savefig(output_path)
    print('图像输入至' + output_path)
    #plt.show()
    plt.close()

def compare_different_batch(model, optimizer, output_path, epoch, 
                            batch_list, TrainDataset):
    """比较不同batch大小对训练带来的影响"""
    output_path = os.path.join(output_path, 'compare_diff_batch.png')
    record = {}
    length = {} 
    for batch_size in batch_list:
        opt_copy = copy.deepcopy(optimizer)
        model_copy = copy.deepcopy(model)
        TrainDataloader = DataLoader(TrainDataset, batch_size=int(batch_size), 
                                     shuffle=True)
        _, loss_history = train(TrainDataloader, opt_copy, model_copy, epoch)
        record[str(batch_size)] = loss_history
        length[str(batch_size)] = len(loss_history)
    plt.figure(figsize=(6, 4))
    plt.xlabel('step') 
    plt.ylabel('loss')
    plt.grid(True)
    for batch_size in record.keys():
        x = np.array(range(1, length[batch_size]+1))
        plt.plot(x, record[batch_size], linewidth=1.0, label=batch_size, 
                 alpha=0.4)
        color = plt.gca().lines[-1].get_color()
        ewm = pd.Series(record[batch_size]).ewm(alpha=0.2).mean()
        plt.plot(x, ewm, color=color)
        plt.axhline(y=ewm.iat[-1], color=color, linestyle='--', alpha=0.4)
    plt.legend()
    plt.title('Loss under Different Batch Size')
    plt.savefig(output_path)
    print('图像输入至' + output_path)
    #plt.show()
    plt.close()


def compare_different_model(model_list, optimizer_list, output_path, epoch, 
                            TrainDataloader):
    """验证不同的模型所带来的影响"""
    output_path = os.path.join(output_path, 'compare_diff_model.png')
    length = 0
    optnum = len(optimizer_list)
    subcol = 4
    count = 1
    if optnum % 4 == 0:
        subrow = optnum / 4
    else:
        subrow = optnum // 4 + 1
        flag = optnum % 4
    plt.figure(figsize=(14, 4 * subrow))
    plt.title('Loss for Different Model')
    ax = plt.gca()
    ax.axis('off')
    for opt in optimizer_list:
        plt.subplot(subrow, subcol, count)
        plt.grid(True)
        plt.yticks(rotation=45)
        for model in model_list:
            opt_model = copy.deepcopy(opt)
            model_copy = copy.deepcopy(model)
            _, loss_history = train(TrainDataloader, opt_model, model_copy, epoch)
            if length == 0:
                length = len(loss_history)
                x = np.array(range(1, length+1))
            plt.plot(x, loss_history, linewidth=1.0, label=str(model), 
                     alpha=0.4)
            color = plt.gca().lines[-1].get_color()
            ewm = pd.Series(loss_history).ewm(alpha=0.2).mean()
            plt.axhline(y=ewm.iat[-1], color=color, linestyle='--', alpha=0.4)
            plt.plot(x, ewm, color=color)
        plt.legend()
        plt.title(f'{str(opt)}', y=-0.2)
        count += 1
    for _ in range(flag):
        plt.subplot()
    plt.tight_layout()
    plt.savefig(output_path)
    print('图像输入至' + output_path)
    #plt.show()
    plt.close() 

def compare_different_learning_rate(model, lr_list, opt_class_list, output_path, 
                                    epoch, TrainDataloader, init):
    """比较不同的学习率带来的影响"""
    output_path = os.path.join(output_path, 'compare_diff_lr.png')
    length = 0
    optnum = len(opt_class_list)
    subcol = 4
    count = 1
    model = copy.deepcopy(model)
    if optnum % 4 == 0:
        subrow = int(optnum / 4)
        flag = 0
    else:
        subrow = optnum // 4 + 1
        flag = optnum % 4
    plt.figure(figsize=(14, 4 * subrow))
    #plt.title('Loss for Different Learning Rate')
    ax = plt.gca()
    ax.axis('off')
    for opt in opt_class_list:
        plt.subplot(subrow, subcol, count)
        plt.grid(True)
        plt.yticks(rotation=45)
        for lr in lr_list:
            opt_lr = opt(copy.deepcopy(init), lr)
            _, loss_history = train(TrainDataloader, opt_lr, model, epoch)
            if length == 0:
                length = len(loss_history)
                x = np.array(range(1, length+1))
            #plt.plot(x, loss_history, linewidth=1.0, label=str(lr), 
                     #alpha=0.4)    
            #color = plt.gca().lines[-1].get_color()
            ewm = pd.Series(loss_history).ewm(alpha=0.2).mean()
            plt.plot(x, ewm, label=str(lr))
            color = plt.gca().lines[-1].get_color()
            plt.axhline(y=ewm.iat[-1], color=color, linestyle='--', alpha=0.4)
        plt.legend()
        plt.title(f'{str(opt_lr)}', y=-0.2)
        count += 1
    for _ in range(flag):
        plt.subplot()
    plt.tight_layout()
    plt.savefig(output_path)
    print('图像输入至' + output_path)
    #plt.show()
    plt.close()

def argslist(value):
    return [item for item in value.split(',')]

def convert_to_int_list(*args):
    for arg in args:
        arg = [int(x) for x in arg[0]] 

        
def parse_args(args):
    parser = argparse.ArgumentParser(description='观察不同变量对逻辑斯蒂回归模型的影响')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_list', type=argslist, default=[512, 256, 1024],
                        help='第一个元素将被用作其他对比的默认Batch_size')
    parser.add_argument('--optimizer_list', type=argslist, 
                        default=['AdaDelta', 'SGD', 'Momentum', \
                                 'Nesterov', 'AdaGrad', 'RMSProp', \
                                 'Adam', 'Adafactor'], help='第一个元素将用作默认优化器')
    parser.add_argument('--output_file', type=str, default='Store_Plot')
    parser.add_argument('--model_list', type=argslist, 
                        default=['VanillaLogReg', 'VanillaLogRegNorm'],
                        help='第一个元素将被用作默认模型')
    parser.add_argument('--lr_list', type=argslist, default=[0.001, 0.01, 0.05])
    parser.add_argument('--datapath', type=str, default='spotify_songs.csv')
    parser.add_argument('--norm_weight', type=int, default=1)
    args = parser.parse_args(args)
    return args

def main(args):
    args.lr_list = [float(lr) for lr in args.lr_list]
    args.batch_list = [int(batch) for batch in args.batch_list]
    dataset = SpotifyDataset(args.datapath)
    print('完成数据集导入')
    train_data, valid_data = random_split(dataset, lengths=[0.9, 0.1])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_list[0],
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_list[0],
                                  shuffle=True)
    os.makedirs(args.output_file, exist_ok=True)
    optmap = {'SGD': Vanilla_SGD,
              'Momentum': Momentum_SGD,
              'Nesterov': Nesterov_SGD,
              'AdaGrad': AdaGrad_SGD,
              'RMSProp':RMSProp_SGD,
              'AdaDelta': AdaDelta_SGD,
              'Adam': Adam_SGD,
              'Adafactor': AdaFactor_SGD}

    modelmap = {'VanillaLogReg': VanillaLogReg,
                'VanillaLogRegNorm': VanillaLogRegNorm}
    opt_class_list = [optmap[opt] for opt in args.optimizer_list]
    model_class_list = [modelmap[model] for model in args.model_list]
    theta_size = train_data[0][0].shape[0]
    theta_init = torch.rand(theta_size, 6, dtype=torch.float64)
    model_list = [model(theta_init) if 'norm' not in str(model).lower() else 
                  model(theta_init, args.norm_weight) 
                  for model in model_class_list]
    opt_list = [opt(theta_init, args.lr_list[0]) for opt in opt_class_list]
    args.lr_list = np.sort(args.lr_list)
    args.batch_list = np.sort(args.batch_list)
    print('开始绘图')
    compare_different_optimizer(model_list[0], opt_list, args.output_file, 
                                args.epoch, train_dataloader, valid_dataloader)
    #compare_different_batch(model_list[0], opt_list[0], args.output_file, args.epoch,
    #                        args.batch_list, train_data)
    #compare_different_model(model_list, opt_list, args.output_file, args.epoch,
    #                        train_dataloader)
    #compare_different_learning_rate(model_list[0], args.lr_list, opt_class_list, 
                                    #args.output_file, args.epoch, train_dataloader,
                                    #theta_init)
    print('完成绘图') 
    
if __name__ == '__main__':
    print('Plot Start')
    args = parse_args(None)
    main(args)
