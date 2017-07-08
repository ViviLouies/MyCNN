close all;clear;clc;
%网络定义
cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 6, 'kernelsize', [5,5], 'stride', 1,'function','relu') %convolution layer
    struct('type', 'pool', 'scale', 2, 'method', 'max','weight',0,'function',0) %pool layer（默认有偏置bias，权值和传递函数可以选择）
    struct('type', 'conv', 'outputmaps', 16, 'kernelsize', [3,3], 'stride', 1,'function','relu') %convolution layer
    struct('type', 'bn') %bn layer
    struct('type', 'conv', 'outputmaps', 48, 'kernelsize', [3,3], 'stride', 1) %convolution layer
    struct('type', 'pool', 'scale', 2, 'method', 'mean','weight',0,'function',0) %pool layer
    struct('type', 'conv', 'outputmaps', 120, 'kernelsize', [2,2], 'stride', 1,'function','relu') %convolution layer
    struct('type', 'bn') %bn layer
    %struct('type', 'fc', 'outputmaps', 120) %full connecting layer
    struct('type', 'fc', 'outputmaps', 64) %full connecting layer
    struct('type', 'loss','function', 'softmax') %loss layer
    };
%训练参数定义
opts.alpha = 0.5;   %初始学习率
opts.momentum = 0.5;  %初始动量项权值
opts.batchsize = 1000; %批大小
opts.numepochs = 30;  %迭代次数

%导入数据和类标(one vs all)
load mnist_uint8.mat; %加载数据集 
train_data = double(reshape(train_x',28,28,60000))/255;
test_data = double(reshape(test_x',28,28,10000))/255;
train_label = double(train_y');
test_label = double(test_y');
% 建立CNN网络
inputSize = [28,28]; %输入图片尺寸
outputSize = 10; %输出类别数目
cnn= cnnSetup(cnn, inputSize, outputSize);
% 训练网络
cnn = cnnTrain(cnn,train_data,train_label,opts);
% 测试网络
[accuracy, index] = cnnTest(cnn,test_data,test_label);