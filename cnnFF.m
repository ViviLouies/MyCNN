function net = cnnFF(net, data)
%CNNǰ�򴫲�
for layer = 1 : numel(net.layers)   %����ÿ��
    switch net.layers{layer}.type
        case 'input'  %�����
            net.layers{layer}.a{1} = data; %����ĵ�һ������������ݣ������˶��ѵ��ͼ�񣬵�ֻ��һ������ͼ
        case 'conv'  %�����
            net.layers{layer}.a = cnnConvolve(net.layers{layer-1}.a,net.layers{layer}); %���ﲻ���Ǿֲ��������
        case 'pool'  %�ػ���
            [net.layers{layer}.downsample,net.layers{layer}.a,net.layers{layer}.maxPos] = cnnPool(net.layers{layer-1}.a,net.layers{layer});
        case 'bn'
            net.layers{layer} = batch_normlization(net.layers{layer-1}.a,net.layers{layer});
        case 'fc'  %ȫ���Ӳ�
            fc_in = [];
            if strcmp(net.layers{layer-1}.type, 'conv') || strcmp(net.layers{layer-1}.type, 'pool') || strcmp(net.layers{layer-1}.type, 'bn')%���ȫ���Ӳ��ǰһ���Ǿ���㡢�ػ����batch normalization��mapsize��С����[1,1]����Ҫ��������
                for map = 1:numel(net.layers{layer-1}.a) %ǰһ���outputmaps��Ŀ����ȫ���Ӳ��inputmaps��Ŀ
                    [m,n,datanum] = size(net.layers{layer-1}.a{map});
                    fc_in = [fc_in; reshape(net.layers{layer-1}.a{map},m*n,datanum)]; %��ǰһ������mapsչ��������
                end
            else
                fc_in = net.layers{layer-1}.a; %���ȫ���Ӳ��ǰһ����ȫ���Ӳ㣨mapsize��С��[1,1]�������ý�ǰһ���outputmapչ������
                datanum = size(net.layers{layer-1}.a,2);
            end
            net.layers{layer}.input = fc_in; %����ǰһ�������������������ʽ��
            if strcmp(net.layers{layer}.function, 'sigmoid')
                net.layers{layer}.a = sigmoid(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %����������
            elseif  strcmp(net.layers{layer}.function, 'tanh')
                net.layers{layer}.a = tanh(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %����������
            elseif strcmp(net.layers{layer}.function, 'relu')
                net.layers{layer}.a = relu(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %����������
            else
                error('Unknown function of full connection layer!')
            end
        case 'loss' %��ʧ��,�����һ��
            fc_in = [];
            if strcmp(net.layers{layer-1}.type, 'conv') || strcmp(net.layers{layer-1}.type, 'pool') %�����ʧ���ǰһ���Ǿ�����ػ��㣨mapsize��С����[1,1]����Ҫ��������
                for map = 1:numel(net.layers{layer-1}.a) %ǰһ���outputmaps��Ŀ����ȫ���Ӳ��inputmaps��Ŀ
                    [m,n,datanum] = size(net.layers{layer-1}.a{map});
                    fc_in = [fc_in; reshape(net.layers{layer-1}.a{map},m*n,datanum)]; %��ǰһ������mapsչ��������
                end
            else
                fc_in = net.layers{layer-1}.a; %�����ʧ���ǰһ����ȫ���Ӳ㣨mapsize��С��[1,1]�������ý�ǰһ���outputmapչ������
                datanum = size(net.layers{layer-1}.a,2);
            end
            net.layers{layer}.input = fc_in; %����ǰһ�㣨�����ڶ��㣩������������������ʽ��
            if strcmp(net.layers{layer}.function, 'sigmoid')
                net.layers{layer}.a = sigmoid(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %�������label
            elseif strcmp(net.layers{layer}.function, 'tanh')
                net.layers{layer}.a = tanh(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %�������label
            elseif strcmp(net.layers{layer}.function, 'softmax')
                temp = net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum);
                M = bsxfun(@minus,temp,max(temp, [], 1)); %max(theta*data, [], 1)������е����ֵ�����һ��������
                M = exp(M);
                net.layers{layer}.a = bsxfun(@rdivide, M, sum(M));  %�������label
            else
                error('Unknown function of loss layer!')
            end
    end
end
