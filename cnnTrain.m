function net = cnnTrain(net, data, label, opts)
% net: 网络结构
% data：训练数据
% label：训练数据对应标签
% opts：网络训练参数，包括：
% opts.batchsize 批大小
% opts.numepochs 迭代次数
% opts.alpha 学习率
% opts.momentum 动量项

data_num = size(data, 3);  %训练样本个数
disp(['num of data = ' num2str(data_num)]);
batche_num = data_num / opts.batchsize;  %batchenum表示每次迭代（批）中所选取的训练样本数
if rem(batche_num, 1) ~= 0
    error('batchenum is not an integer!!');
end
interval = ceil(opts.numepochs/4) + 1;
inc = 1;
momentum = [0.9,0.95,0.99]; %动量项每迭代interval次数时更新一次
cost = zeros(opts.numepochs,1);  %程序运行代价
time = zeros(opts.numepochs,1);  %程序运行时间
loss = []; %网络的训练误差
for epoch = 1 : opts.numepochs  %对于每次迭代
    disp(['epoch ' num2str(epoch) '/' num2str(opts.numepochs) ':']);
    tic;  % 计时
    if rem(epoch,interval)==0
        opts.momentum = momentum(inc); %每interval次迭代更新一次动量项
        inc= inc + 1;
    end
    if rem(epoch,10)==0
        opts.alpha = opts.alpha * 0.2; %每10次迭代更新一次学习速率
    end
    index = randperm(data_num);  %打散样本
    for batch = 1 : batche_num
        %依次取出每一次训练用的样本
        batch_x = data(:, :, index((batch - 1) * opts.batchsize + 1 : batch * opts.batchsize));
        batch_y = label(:,index((batch - 1) * opts.batchsize + 1 : batch * opts.batchsize));
        %CNN前向计算
        net = cnnFF(net, batch_x);
        %CNN误差反向传播
        net = cnnBP(net, batch_y);
        %网络权值更新
        net = cnnWeightUpdate(net, opts);
        %代价函数值，也就是均方误差值，依次累加
        if isempty(loss) %对于第一次循环
            loss(1) = net.loss;
        end
        loss(end+1) = 0.99*loss(end) + 0.01*net.loss;
    end
    cost(epoch,1) = loss(end);
    time(epoch,1) = toc;
    fprintf('cost = %f \n',cost(epoch,1));
    disp(['runing time：',num2str(toc),'s']);
end
plot(loss);
end