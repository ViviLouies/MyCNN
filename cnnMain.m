close all;clear;clc;
%���綨��
cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 6, 'kernelsize', [5,5], 'stride', 1,'function','relu') %convolution layer
    struct('type', 'pool', 'scale', 2, 'method', 'max','weight',0,'function',0) %pool layer��Ĭ����ƫ��bias��Ȩֵ�ʹ��ݺ�������ѡ��
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
%ѵ����������
opts.alpha = 0.5;   %��ʼѧϰ��
opts.momentum = 0.5;  %��ʼ������Ȩֵ
opts.batchsize = 1000; %����С
opts.numepochs = 30;  %��������

%�������ݺ����(one vs all)
load mnist_uint8.mat; %�������ݼ� 
train_data = double(reshape(train_x',28,28,60000))/255;
test_data = double(reshape(test_x',28,28,10000))/255;
train_label = double(train_y');
test_label = double(test_y');
% ����CNN����
inputSize = [28,28]; %����ͼƬ�ߴ�
outputSize = 10; %��������Ŀ
cnn= cnnSetup(cnn, inputSize, outputSize);
% ѵ������
cnn = cnnTrain(cnn,train_data,train_label,opts);
% ��������
[accuracy, index] = cnnTest(cnn,test_data,test_label);