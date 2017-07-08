%%CNN网络的数值梯度核对
clear;clc;
%网络定义
cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 2, 'kernelsize', [5,5], 'stride', 1,'function','tanh') %convolution layer
    struct('type', 'pool', 'scale', 2, 'method', 'max','weight',0,'function',0) %pool layer（默认有偏置bias，权值和传递函数可以选择）
    struct('type', 'conv', 'outputmaps', 4, 'kernelsize', [3,3], 'stride', 1,'function','tanh') %convolution layer
    struct('type', 'bn') %bn layer
    struct('type', 'conv', 'outputmaps', 8, 'kernelsize', [3,3], 'stride', 1) %convolution layer
    struct('type', 'pool', 'scale', 2, 'method', 'mean','weight',0,'function',0) %pool layer
    struct('type', 'conv', 'outputmaps', 16, 'kernelsize', [2,2], 'stride', 1,'function','tanh') %convolution layer
    struct('type', 'bn') %bn layer
    %struct('type', 'fc', 'outputmaps', 120) %full connecting layer
    struct('type', 'fc', 'outputmaps', 8) %full connecting layer
    struct('type', 'loss','function', 'softmax') %loss layer
    %{
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 4, 'kernelsize', [3,3], 'stride', 1,'function','sigmoid') %convolution layer
    struct('type', 'pool', 'scale', 2, 'method', 'mean','weight',1,'function',0) %subsampling layer
    struct('type', 'conv', 'outputmaps', 8, 'kernelsize', [2,2], 'stride', 2,'function','relu') %convolution layer
    %struct('type', 'pool', 'scale', 2, 'method', 'mean','weight',0,'function',1) %subsampling layer
    %struct('type', 'conv', 'outputmaps', 20, 'kernelsize', [2,2], 'stride', 1) %convolution layer
    struct('type', 'bn');
    struct('type', 'fc', 'outputmaps', 12, 'function','tanh') %full connecting layer
    struct('type', 'fc', 'outputmaps', 6, 'function','relu') %full connecting layer
    struct('type', 'loss','function', 'softmax') %loss layer
    %}
    };
%注意：如果卷积层使用的是relu激活函数（在0点不可导！！），则在计算卷积层的数值梯度时，可能不准确
%若x<0,则f(x)=0，其解析梯度f'(x)=0;
%若h>0,且正好h>|x|,则f(x+h)=x+h>0（越过不可导点0）,其数值梯度f'(x+h)=1>0,与解析梯度不一致（f(x-h)同理）
%但是，在不可导点附近的x只是少数，如果sigmoid或者tanh激活函数的梯度检验通过，且relu激活函数的大部分数值梯度检验通过（有时会全部通过），则梯度计算没有问题

%训练参数定义
opts.alpha = 0.01;   %学习率
opts.momentum = 0.9;  %动量项权值
opts.batchsize = 100; %批大小
opts.numepochs = 20;  %迭代次数
%训练数据
data = rand(28,28,4);
label = eye(4,4);
inputSize = [28,28];
outputSize = 4;
cnn= cnnSetup(cnn, inputSize, outputSize);
cnn = cnnFF(cnn,data);
cnn = cnnBP(cnn,label);
cnn = cnnWeightUpdate(cnn, opts);%一定要跟新一次权重才能确保梯度算法运行正确，避免偶然性
cnn = cnnFF(cnn,data);
cnn = cnnBP(cnn,label);
cnnNumGradCheck(cnn,data,label);