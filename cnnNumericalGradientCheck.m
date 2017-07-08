%%CNN�������ֵ�ݶȺ˶�
clear;clc;
%���綨��
cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 2, 'kernelsize', [5,5], 'stride', 1,'function','tanh') %convolution layer
    struct('type', 'pool', 'scale', 2, 'method', 'max','weight',0,'function',0) %pool layer��Ĭ����ƫ��bias��Ȩֵ�ʹ��ݺ�������ѡ��
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
%ע�⣺��������ʹ�õ���relu���������0�㲻�ɵ������������ڼ����������ֵ�ݶ�ʱ�����ܲ�׼ȷ
%��x<0,��f(x)=0��������ݶ�f'(x)=0;
%��h>0,������h>|x|,��f(x+h)=x+h>0��Խ�����ɵ���0��,����ֵ�ݶ�f'(x+h)=1>0,������ݶȲ�һ�£�f(x-h)ͬ��
%���ǣ��ڲ��ɵ��㸽����xֻ�����������sigmoid����tanh��������ݶȼ���ͨ������relu������Ĵ󲿷���ֵ�ݶȼ���ͨ������ʱ��ȫ��ͨ���������ݶȼ���û������

%ѵ����������
opts.alpha = 0.01;   %ѧϰ��
opts.momentum = 0.9;  %������Ȩֵ
opts.batchsize = 100; %����С
opts.numepochs = 20;  %��������
%ѵ������
data = rand(28,28,4);
label = eye(4,4);
inputSize = [28,28];
outputSize = 4;
cnn= cnnSetup(cnn, inputSize, outputSize);
cnn = cnnFF(cnn,data);
cnn = cnnBP(cnn,label);
cnn = cnnWeightUpdate(cnn, opts);%һ��Ҫ����һ��Ȩ�ز���ȷ���ݶ��㷨������ȷ������żȻ��
cnn = cnnFF(cnn,data);
cnn = cnnBP(cnn,label);
cnnNumGradCheck(cnn,data,label);