function net = cnnBP(net,label)
%% 计算残差（灵敏度）
lambda = 1e-4; %softmax层的权重衰减系数
layer_num = numel(net.layers); %网络层数
for layer = layer_num : -1 : 1
    switch net.layers{layer}.type
        %=======================%如果是损失层，则先计算loss，再反向传播%===================%
        case 'loss'
            net.layers{layer}.error = net.layers{layer}.a - label; %实际输出与期望输出之间的误差
            if strcmp(net.layers{layer}.function, 'sigmoid')
                net.loss =  1/2* sum(net.layers{layer}.error(:) .^ 2) / size(net.layers{layer}.a, 2);  %代价函数，采用均方误差函数作为代价函数
                net.layers{layer}.delta = net.layers{layer}.error .* (net.layers{layer}.a .* (1 - net.layers{layer}.a)); %输出层残差sigmoid传递函数
            elseif strcmp(net.layers{layer}.function, 'tanh')
                net.loss =  1/2* sum(net.layers{layer}.error(:) .^ 2) / size(net.layers{layer}.a, 2);  %代价函数，采用均方误差函数作为代价函数
                net.layers{layer}.delta = net.layers{layer}.error .* (1 - (net.layers{layer}.a).^2); %输出层残差tanh传递函数
            elseif strcmp(net.layers{layer}.function, 'softmax')
                net.loss = -1/size(net.layers{layer}.a, 2) * label(:)' * log(net.layers{layer}.a(:)) + lambda/2 * sum(net.layers{layer}.w(:) .^ 2);  %softmax损失函数，加入权重衰减处理参数冗余
                net.layers{layer}.delta = net.layers{layer}.a - label;  %softmax层的灵敏度
            end
            %若损失层的前一层是卷积层、池化层或batch normalization层，其间隐含着一层光栅层（矢量化层，即将卷积或池化的outputmaps拉成向量），需要计算这隐含层的灵敏度
            if strcmp(net.layers{layer-1}.type, 'pool') %若全连接层的前一层是池化层(默认池化层无激活函数)
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %隐含光栅层的灵敏度
                if net.layers{layer-1}.function   %若池化层有sigmoid传递函数，则乘以该层的输出偏导数（即当前全连接层的输入）
                    net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                end
                %反矢量化：将光栅层的灵敏度（列向量）扩展成与前一层池化层的特征map相同的尺寸形式
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %取前一层特征map尺寸
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %前一层的特征map的个数
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            elseif strcmp(net.layers{layer-1}.type, 'bn')  %如果全连接层的前一层是batch normalization层（无激活函数）,直接反矢量化级即可
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %隐含光栅层的灵敏度
                %反矢量化：将光栅层的灵敏度（列向量）扩展成与前一层池化层的特征map相同的尺寸形式
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %取前一层特征map尺寸
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %前一层的特征map的个数
                    z = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                    net.layers{layer-1}.dvar{outputmap,1} = -1/2 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z .* net.layers{layer-1}.z_decent{outputmap,1},3) ./ (net.layers{layer-1}.std{outputmap,1}).^3; %方差导数
                    net.layers{layer-1}.dmean{outputmap,1} = -1 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z,3) ./ net.layers{layer-1}.std{outputmap,1}...
                        - 2 .* net.layers{layer-1}.dvar{outputmap,1} .* mean(net.layers{layer-1}.z_decent{outputmap,1},3);     %均值导数
                    net.layers{layer-1}.delta{outputmap,1} = net.layers{layer-1}.gamma{outputmap,1} .* z ./ repmat(net.layers{layer-1}.std{outputmap,1},[1,1,data_num])...
                        + 2/data_num .* repmat(net.layers{layer-1}.dvar{outputmap,1},[1,1,data_num]) .* net.layers{layer-1}.z_decent{outputmap,1}...
                        + 1/data_num .* repmat(net.layers{layer-1}.dmean{outputmap,1},[1,1,data_num]);     %残差
                    net.layers{layer-1}.dgamma{outputmap,1} = sum(sum(sum(z .* net.layers{layer-1}.z_norm{outputmap,1}))) ./ data_num;  %gamma偏导数
                    net.layers{layer-1}.dbeta{outputmap,1} = sum(z(:)) ./ data_num;  %beta偏导数
                end
            elseif strcmp(net.layers{layer-1}.type, 'conv')  %如果全连接层的前一层是卷积层（有激活函数）,则残差要乘以前一层的输出偏导数
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta; %隐含光栅层的灵敏度
                switch net.layers{layer-1}.function
                    case 'sigmoid'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                    case 'tanh'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (1 - (net.layers{layer}.input).^2);
                    case 'relu'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* double(net.layers{layer}.input>0.0);
                end
                %反矢量化：将光栅层的灵敏度（列向量）扩展成与前一层卷积层的特征map相同的尺寸形式
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %取前一层特征map尺寸
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %前一层的特征map的个数
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            end
            %=======================%如果是全连接层，则运行传统的BP算法%===================%
        case 'fc'
            %计算当前全连接层的灵敏度
            if strcmp(net.layers{layer}.function, 'sigmoid')
            net.layers{layer}.delta = (net.layers{layer+1}.w' * net.layers{layer+1}.delta) .* (net.layers{layer}.a .* (1 - net.layers{layer}.a)); 
            elseif strcmp(net.layers{layer}.function, 'tanh')
                net.layers{layer}.delta = (net.layers{layer+1}.w' * net.layers{layer+1}.delta) .* (1 - (net.layers{layer}.a).^2);
            elseif strcmp(net.layers{layer}.function, 'relu')
                net.layers{layer}.delta = (net.layers{layer+1}.w' * net.layers{layer+1}.delta) .* double(net.layers{layer}.a>0.0);
            end
            %若该全连接层的前一层是卷积层、池化层或batch normalization层，其间隐含着一层光栅层（矢量化层，即将卷积或池化的outputmaps拉成向量），需要计算这隐含层的灵敏度
            if strcmp(net.layers{layer-1}.type, 'pool') %若全连接层的前一层是池化层(默认池化层无激活函数)
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %隐含层的灵敏度
                if net.layers{layer-1}.function   %若池化层有sigmoid传递函数，则乘以该层的输出偏导数（即当前全连接层的输入）
                    net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                end
                %反矢量化：将光栅层的灵敏度（列向量）扩展成与前一层池化层的特征map相同的尺寸形式
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %取前一层特征map尺寸
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %前一层的特征map的个数
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            elseif strcmp(net.layers{layer-1}.type, 'bn')  %如果全连接层的前一层是batch normalization层（无激活函数）,直接反矢量化级即可
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %隐含光栅层的灵敏度
                %反矢量化：将光栅层的灵敏度（列向量）扩展成与前一层池化层的特征map相同的尺寸形式
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %取前一层特征map尺寸
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %前一层的特征map的个数
                    z = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                    net.layers{layer-1}.dvar{outputmap,1} = -1/2 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z .* net.layers{layer-1}.z_decent{outputmap,1},3) ./ (net.layers{layer-1}.std{outputmap,1}).^3; %方差导数
                    net.layers{layer-1}.dmean{outputmap,1} = -1 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z,3) ./ net.layers{layer-1}.std{outputmap,1}...
                        - 2 .* net.layers{layer-1}.dvar{outputmap,1} .* mean(net.layers{layer-1}.z_decent{outputmap,1},3);     %均值导数
                    net.layers{layer-1}.delta{outputmap,1} = net.layers{layer-1}.gamma{outputmap,1} .* z ./ repmat(net.layers{layer-1}.std{outputmap,1},[1,1,data_num])...
                        + 2/data_num .* repmat(net.layers{layer-1}.dvar{outputmap,1},[1,1,data_num]) .* net.layers{layer-1}.z_decent{outputmap,1}...
                        + 1/data_num .* repmat(net.layers{layer-1}.dmean{outputmap,1},[1,1,data_num]);     %残差
                    net.layers{layer-1}.dgamma{outputmap,1} = sum(sum(sum(z .* net.layers{layer-1}.z_norm{outputmap,1}))) ./ data_num;  %gamma偏导数
                    net.layers{layer-1}.dbeta{outputmap,1} = sum(z(:)) ./ data_num;  %beta偏导数
                end
            elseif strcmp(net.layers{layer-1}.type, 'conv')  %如果全连接层的前一层是卷积层（有激活函数）,则残差要乘以前一层的输出偏导数
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta; %隐含层的灵敏度
                switch net.layers{layer-1}.function
                    case 'sigmoid'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                    case 'tanh'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (1 - (net.layers{layer}.input).^2);
                    case 'relu'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* double(net.layers{layer}.input>0.0);
                end
                %反矢量化：将光栅层的灵敏度（列向量）扩展成与前一层卷积层的特征map相同的尺寸形式
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %取前一层特征map尺寸
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %前一层的特征map的个数
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            end
            %====%如果是卷积层，则进行上采样(采用maxPos位置标记来统一最大池化和平均池化的灵敏度计算)%====%
        case 'conv' %卷积层后面只可能是卷积层、池化层或batch normalization层，一般不会直接连全连接层或损失层（就算后面连接了这两种类型，其灵敏度也在计算光栅层的时候就算好了）
            if strcmp(net.layers{layer+1}.type, 'pool') %若卷积层后面是池化层
                for outputmap = 1 : net.layers{layer}.outputmaps  %该层特征map的个数
                    switch net.layers{layer}.function
                        case 'sigmoid'
                            net.layers{layer}.delta{outputmap,1} = net.layers{layer}.a{outputmap} .* (1 - net.layers{layer}.a{outputmap})...
                                .* (expand(net.layers{layer+1}.delta{outputmap,1}, [net.layers{layer+1}.scale,net.layers{layer+1}.scale,1]) .* net.layers{layer+1}.maxPos{outputmap});
                        case 'tanh'
                            net.layers{layer}.delta{outputmap,1} = (1 - (net.layers{layer}.a{outputmap}).^2)...
                                .* (expand(net.layers{layer+1}.delta{outputmap,1}, [net.layers{layer+1}.scale,net.layers{layer+1}.scale,1]) .* net.layers{layer+1}.maxPos{outputmap});
                        case 'relu'
                            net.layers{layer}.delta{outputmap,1} = double(net.layers{layer}.a{outputmap}>0.0)...
                                .* (expand(net.layers{layer+1}.delta{outputmap,1}, [net.layers{layer+1}.scale,net.layers{layer+1}.scale,1]) .* net.layers{layer+1}.maxPos{outputmap});
                    end
                    if net.layers{layer+1}.weight || net.layers{layer+1}.function %若后面的池化层有权重或传递函数(默认包含权重),则根据残差传递函数还要乘上池化层的权重
                        net.layers{layer}.delta{outputmap,1} = net.layers{layer}.delta{outputmap,1} * net.layers{layer+1}.w{outputmap};
                    end
                end
            elseif strcmp(net.layers{layer+1}.type, 'bn') %若卷积层后面是batch normalization层(和pool层相似处理)
                for outputmap = 1 : net.layers{layer}.outputmaps  %该层特征map的个数
                    switch net.layers{layer}.function
                        case 'sigmoid'
                            net.layers{layer}.delta{outputmap,1} = net.layers{layer}.a{outputmap} .* (1 - net.layers{layer}.a{outputmap}) .* net.layers{layer+1}.delta{outputmap,1};      
                        case 'tanh'
                            net.layers{layer}.delta{outputmap,1} = (1 - (net.layers{layer}.a{outputmap}).^2) .* net.layers{layer+1}.delta{outputmap,1};
                        case 'relu'
                            net.layers{layer}.delta{outputmap,1} = double(net.layers{layer}.a{outputmap}>0.0) .* net.layers{layer+1}.delta{outputmap,1}; 
                    end
                end
            elseif strcmp(net.layers{layer+1}.type, 'conv') %若卷积层后面是卷积层
                for inputmap = 1 : net.layers{layer}.outputmaps %第layer层特征map的个数
                    z = zeros(size(net.layers{layer}.a{1}));
                    for outputmap = 1 : net.layers{layer+1}.outputmaps %第layer+1层特征map的个数
                        %任意步长时的情形
                        padMap = map_padding(net.layers{layer+1}.delta{outputmap,1},net.layers{layer+1}.mapsize,net.layers{layer+1}.kernelsize,net.layers{layer+1}.stride);
                        switch net.layers{layer}.function
                            case 'sigmoid'
                                z = z + net.layers{layer}.a{inputmap} .* (1 - net.layers{layer}.a{inputmap})...
                                    .* convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid'); %由于convn会自动旋转卷积核，故这里不再旋转
                            case 'tanh'
                                z = z + (1 - (net.layers{layer}.a{inputmap}).^2)...
                                    .* convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid'); %由于convn会自动旋转卷积核，故这里不再旋转
                            case 'relu'
                                z = z + double(net.layers{layer}.a{inputmap}>0.0)...
                                    .* convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid'); %由于convn会自动旋转卷积核，故这里不再旋转)
                        end
                    end
                    net.layers{layer}.delta{inputmap,1} = z;
                end
            end
            %=======================%如果是池化层，则进行下采样%===================%
        case 'pool'
            if strcmp(net.layers{layer+1}.type, 'fc') || strcmp(net.layers{layer+1}.type, 'loss')
                continue;  %默认池化层后一层是卷积层,若是全连接层或损失层，其灵敏度在光栅层已计算，此处可以省略
            else
                for inputmap = 1 : net.layers{layer}.outputmaps %第layer层特征map的个数,和第layer-1卷积层的特征map数目相同
                    z = zeros(size(net.layers{layer}.a{1}));
                    for outputmap = 1 : net.layers{layer+1}.outputmaps %第layer+1层特征map的个数
                        %任意步长时的情形
                        padMap = map_padding(net.layers{layer+1}.delta{outputmap,1},net.layers{layer+1}.mapsize,net.layers{layer+1}.kernelsize,net.layers{layer+1}.stride);
                        z = z + convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid');
                        %由于convn会自动旋转卷积核，故这里不再旋转
                    end
                    net.layers{layer}.delta{inputmap,1} = z;
                    if net.layers{layer}.function %若池化层有sigmoid传递函数
                        net.layers{layer}.delta{inputmap,1} = z .* net.layers{layer}.a{inputmap} .* (1 - net.layers{layer}.a{inputmap});
                    end
                end
            end
        %=======================%如果是batch normalization层%===================%
        case 'bn' %batch normalization层一般后接卷积层，且不和池化层连续出现
            if strcmp(net.layers{layer+1}.type, 'fc') || strcmp(net.layers{layer+1}.type, 'loss')
                continue;  %默认batch normalization层后一层是卷积层,若是全连接层或损失层，其灵敏度在光栅层已计算，此处可以省略
            elseif strcmp(net.layers{layer+1}.type, 'conv') %(和pool层相似处理)
                 for inputmap = 1 : net.layers{layer}.outputmaps %第layer层特征map的个数,和第layer-1卷积层的特征map数目相同
                    z = zeros(size(net.layers{layer}.a{1}));
                    for outputmap = 1 : net.layers{layer+1}.outputmaps %第layer+1层特征map的个数
                        %任意步长时的情形
                        padMap = map_padding(net.layers{layer+1}.delta{outputmap,1},net.layers{layer+1}.mapsize,net.layers{layer+1}.kernelsize,net.layers{layer+1}.stride);
                        z = z + convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid');
                        %由于convn会自动旋转卷积核，故这里不再旋转
                    end
                    data_num = size(net.layers{layer}.a{1},3); %mini-batch 数目
                    net.layers{layer}.dvar{inputmap,1} = -1/2 .* net.layers{layer}.gamma{inputmap,1} .* sum(z .* net.layers{layer}.z_decent{inputmap,1},3) ./ (net.layers{layer}.std{inputmap,1}).^3; %方差导数
                    net.layers{layer}.dmean{inputmap,1} = -1 .* net.layers{layer}.gamma{inputmap,1} .* sum(z,3) ./ net.layers{layer}.std{inputmap,1}...     
                        - 2 .* net.layers{layer}.dvar{inputmap,1} .* mean(net.layers{layer}.z_decent{inputmap,1},3);     %均值导数
                    net.layers{layer}.delta{inputmap,1} = net.layers{layer}.gamma{inputmap,1} .* z ./ repmat(net.layers{layer}.std{inputmap,1},[1,1,data_num])...
                        + 2/data_num .* repmat(net.layers{layer}.dvar{inputmap,1},[1,1,data_num]) .* net.layers{layer}.z_decent{inputmap,1}...
                        + 1/data_num .* repmat(net.layers{layer}.dmean{inputmap,1},[1,1,data_num]);     %残差
                    net.layers{layer}.dgamma{inputmap,1} = sum(sum(sum(z .* net.layers{layer}.z_norm{inputmap,1}))) ./ data_num;  %gamma偏导数
                    net.layers{layer}.dbeta{inputmap,1} = sum(z(:)) ./ data_num;  %beta偏导数
                 end
            else
                error('situation undefined: %s + %s','bn',net.layers{layer+1}.type);
            end
    end
end

%% 计算梯度
for layer = 2:layer_num
    switch net.layers{layer}.type
        %========================%计算卷积层的导数%===========================%
        case  'conv'
            for outputmap = 1:net.layers{layer}.outputmaps
                for inputmap = 1:net.layers{layer-1}.outputmaps
                    padMap = map_padding(net.layers{layer}.delta{outputmap},net.layers{layer}.mapsize,[1,1],net.layers{layer}.stride);
                    %考虑卷积层的步长，要将卷积层的灵敏度根据卷积步长进行内部扩充（其实这一步就是在按步长进行上采样，用0填充；由于不用进行外部填充，所以将kernelsize设置为[1,1]）
                    z = convn(net.layers{layer-1}.a{inputmap},flipall(padMap), 'valid');
                    %convn会自动旋转卷积核,这里要反旋转回来,flipall函数将矩阵的每一维度(这里是三个维度)都翻转了180度
                    net.layers{layer}.dw{outputmap,inputmap} = z./size(net.layers{layer}.delta{outputmap}, 3);
                end
                net.layers{layer}.db{outputmap,1} = sum(net.layers{layer}.delta{outputmap}(:)) / size(net.layers{layer}.delta{outputmap}, 3);
            end
            %=======================%计算池化层的导数%============================%
        case 'pool'
            for outputmap = 1:net.layers{layer}.outputmaps
                net.layers{layer}.db{outputmap,1} = sum(net.layers{layer}.delta{outputmap}(:)) / size(net.layers{layer}.delta{outputmap}, 3);
                if net.layers{layer}.weight || net.layers{layer}.function %若池化层有权值或传递函数(默认有权值)
                    net.layers{layer}.dw{outputmap,1} = sum(net.layers{layer}.delta{outputmap}(:) .* net.layers{layer}.downsample{outputmap}(:)) / size(net.layers{layer}.delta{outputmap}, 3);
                end
            end
            %==================%计算全连接层的导数%====================%
        case 'fc'
            net.layers{layer}.dw = net.layers{layer}.delta * (net.layers{layer}.input)' / size(net.layers{layer}.delta,2);
            net.layers{layer}.db = mean(net.layers{layer}.delta, 2);
            %==================%计算损失层的导数%====================%
        case 'loss'
            if strcmp(net.layers{layer}.function, 'sigmoid') || strcmp(net.layers{layer}.function, 'tanh')
                net.layers{layer}.dw = net.layers{layer}.delta * (net.layers{layer}.input)' / size(net.layers{layer}.delta,2);
                net.layers{layer}.db = mean(net.layers{layer}.delta, 2);
            elseif strcmp(net.layers{layer}.function, 'softmax')
                net.layers{layer}.dw = net.layers{layer}.delta * (net.layers{layer}.input)' / size(net.layers{layer}.delta, 2) + lambda * net.layers{layer}.w;
                net.layers{layer}.db = mean(net.layers{layer}.delta, 2);
            end
    end
end

end