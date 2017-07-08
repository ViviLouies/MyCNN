function net = cnnFF(net, data)
%CNN前向传播
for layer = 1 : numel(net.layers)   %对于每层
    switch net.layers{layer}.type
        case 'input'  %输入层
            net.layers{layer}.a{1} = data; %网络的第一层就是输入数据，包含了多个训练图像，但只有一个特征图
        case 'conv'  %卷积层
            net.layers{layer}.a = cnnConvolve(net.layers{layer-1}.a,net.layers{layer}); %这里不考虑局部连接情况
        case 'pool'  %池化层
            [net.layers{layer}.downsample,net.layers{layer}.a,net.layers{layer}.maxPos] = cnnPool(net.layers{layer-1}.a,net.layers{layer});
        case 'bn'
            net.layers{layer} = batch_normlization(net.layers{layer-1}.a,net.layers{layer});
        case 'fc'  %全连接层
            fc_in = [];
            if strcmp(net.layers{layer-1}.type, 'conv') || strcmp(net.layers{layer-1}.type, 'pool') || strcmp(net.layers{layer-1}.type, 'bn')%如果全连接层的前一层是卷积层、池化层或batch normalization（mapsize大小不是[1,1]，需要单独处理）
                for map = 1:numel(net.layers{layer-1}.a) %前一层的outputmaps数目，即全连接层的inputmaps数目
                    [m,n,datanum] = size(net.layers{layer-1}.a{map});
                    fc_in = [fc_in; reshape(net.layers{layer-1}.a{map},m*n,datanum)]; %将前一层的输出maps展成列向量
                end
            else
                fc_in = net.layers{layer-1}.a; %如果全连接层的前一层是全连接层（mapsize大小是[1,1]），则不用将前一层的outputmap展成向量
                datanum = size(net.layers{layer-1}.a,2);
            end
            net.layers{layer}.input = fc_in; %保存前一层的输出结果（列向量形式）
            if strcmp(net.layers{layer}.function, 'sigmoid')
                net.layers{layer}.a = sigmoid(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %计算输出结果
            elseif  strcmp(net.layers{layer}.function, 'tanh')
                net.layers{layer}.a = tanh(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %计算输出结果
            elseif strcmp(net.layers{layer}.function, 'relu')
                net.layers{layer}.a = relu(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %计算输出结果
            else
                error('Unknown function of full connection layer!')
            end
        case 'loss' %损失层,即最后一层
            fc_in = [];
            if strcmp(net.layers{layer-1}.type, 'conv') || strcmp(net.layers{layer-1}.type, 'pool') %如果损失层的前一层是卷积层或池化层（mapsize大小不是[1,1]，需要单独处理）
                for map = 1:numel(net.layers{layer-1}.a) %前一层的outputmaps数目，即全连接层的inputmaps数目
                    [m,n,datanum] = size(net.layers{layer-1}.a{map});
                    fc_in = [fc_in; reshape(net.layers{layer-1}.a{map},m*n,datanum)]; %将前一层的输出maps展成列向量
                end
            else
                fc_in = net.layers{layer-1}.a; %如果损失层的前一层是全连接层（mapsize大小是[1,1]），则不用将前一层的outputmap展成向量
                datanum = size(net.layers{layer-1}.a,2);
            end
            net.layers{layer}.input = fc_in; %保存前一层（倒数第二层）的输出结果（列向量形式）
            if strcmp(net.layers{layer}.function, 'sigmoid')
                net.layers{layer}.a = sigmoid(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %计算输出label
            elseif strcmp(net.layers{layer}.function, 'tanh')
                net.layers{layer}.a = tanh(net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum)); %计算输出label
            elseif strcmp(net.layers{layer}.function, 'softmax')
                temp = net.layers{layer}.w * fc_in + repmat(net.layers{layer}.b,1,datanum);
                M = bsxfun(@minus,temp,max(temp, [], 1)); %max(theta*data, [], 1)求出各列的最大值，输出一个行向量
                M = exp(M);
                net.layers{layer}.a = bsxfun(@rdivide, M, sum(M));  %计算输出label
            else
                error('Unknown function of loss layer!')
            end
    end
end
