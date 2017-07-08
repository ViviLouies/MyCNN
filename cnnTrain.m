function net = cnnTrain(net, data, label, opts)
% net: ����ṹ
% data��ѵ������
% label��ѵ�����ݶ�Ӧ��ǩ
% opts������ѵ��������������
% opts.batchsize ����С
% opts.numepochs ��������
% opts.alpha ѧϰ��
% opts.momentum ������

data_num = size(data, 3);  %ѵ����������
disp(['num of data = ' num2str(data_num)]);
batche_num = data_num / opts.batchsize;  %batchenum��ʾÿ�ε�������������ѡȡ��ѵ��������
if rem(batche_num, 1) ~= 0
    error('batchenum is not an integer!!');
end
interval = ceil(opts.numepochs/4) + 1;
inc = 1;
momentum = [0.9,0.95,0.99]; %������ÿ����interval����ʱ����һ��
cost = zeros(opts.numepochs,1);  %�������д���
time = zeros(opts.numepochs,1);  %��������ʱ��
loss = []; %�����ѵ�����
for epoch = 1 : opts.numepochs  %����ÿ�ε���
    disp(['epoch ' num2str(epoch) '/' num2str(opts.numepochs) ':']);
    tic;  % ��ʱ
    if rem(epoch,interval)==0
        opts.momentum = momentum(inc); %ÿinterval�ε�������һ�ζ�����
        inc= inc + 1;
    end
    if rem(epoch,10)==0
        opts.alpha = opts.alpha * 0.2; %ÿ10�ε�������һ��ѧϰ����
    end
    index = randperm(data_num);  %��ɢ����
    for batch = 1 : batche_num
        %����ȡ��ÿһ��ѵ���õ�����
        batch_x = data(:, :, index((batch - 1) * opts.batchsize + 1 : batch * opts.batchsize));
        batch_y = label(:,index((batch - 1) * opts.batchsize + 1 : batch * opts.batchsize));
        %CNNǰ�����
        net = cnnFF(net, batch_x);
        %CNN���򴫲�
        net = cnnBP(net, batch_y);
        %����Ȩֵ����
        net = cnnWeightUpdate(net, opts);
        %���ۺ���ֵ��Ҳ���Ǿ������ֵ�������ۼ�
        if isempty(loss) %���ڵ�һ��ѭ��
            loss(1) = net.loss;
        end
        loss(end+1) = 0.99*loss(end) + 0.01*net.loss;
    end
    cost(epoch,1) = loss(end);
    time(epoch,1) = toc;
    fprintf('cost = %f \n',cost(epoch,1));
    disp(['runing time��',num2str(toc),'s']);
end
plot(loss);
end