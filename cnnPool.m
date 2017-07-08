function [downSample,poolMap,maxPos] = cnnPool(inputMap,poollayer)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%poollayer 池化层参数
%downSample 仅是下采样结果（未经过传递函数，可看成池化的输入）,用于下采样层反向传播过程中的梯度计算
%poolMap 池化结果，还是cell格式
%maxPos 最大值位置记录，用于误差反向传播
inputmaps = numel(inputMap);  %读入inputmaps数目，即outputmaps数目
[height,width,datanum] = size(inputMap{1});  %读入inputmaps大小
stride = poollayer.scale;   %步长默认2
downSample = cell(inputmaps,1); %%预先开辟空间，仅保存下采样结果（便于池化层有权值时权值的更新）
poolMap = cell(inputmaps,1); %预先开辟空间，保存池化结果（下采样 + 运算）
maxPos = cell(inputmaps,1); %预先开辟空间，保存最大值位置
%采用maxPos标记方便后面统一两种情况的灵敏度计算
if strcmp(poollayer.method, 'max') %如果是最大池化
    for map = 1:inputmaps
        downSample{map,1} = zeros(height/poollayer.scale,width/poollayer.scale,datanum); %初始化下采样矩阵为0
        poolMap{map,1} = zeros(height/poollayer.scale,width/poollayer.scale,datanum); %初始化池化矩阵为0
        maxPos{map,1} = zeros(height,width,datanum); %初始化最大值位置矩阵为0
        for row = 1:stride:height
            for col = 1:stride:width
                patch = inputMap{map}(row:row+stride-1,col:col+stride-1,:); %stride*stride patch
                [val,ind] = max(reshape(patch,[stride^2,datanum]));  % 找出最大值及其位置
                downSample{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = val; %保存下采样结果
                poolMap{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = val + poollayer.b{map};  % 加偏置,max pooling,无激活函数
                if poollayer.weight %若池化层有权值
                    poolMap{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = poollayer.w{map} .* val + poollayer.b{map};  % 加权重和偏置,max pooling,无激活函数
                end
                if poollayer.function %若池化层有sigmoid传递函数
                    poolMap{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = sigmoid(poollayer.w{map} .* val + poollayer.b{map});  % 加权重和偏置,max pooling,计算激活函数
                end
                ind_row = rem(ind,stride); %找到最大值索引对应的行坐标(共stride^2个位置)
                ind_row(ind_row==0) = stride; %stride的倍数取余后为0，应加回去
                ind_col = round(ind/stride); %找到最大值索引对应的列坐标
                for i = 1:datanum
                    maxPos{map,1}(row + ind_row(i) - 1, col + ind_col(i) - 1, i) = 1; %推出最大值位置在原图中的相应位置，置为1
                end
            end
        end
    end
elseif strcmp(poollayer.method, 'mean') %如果是平均池化
    for map = 1:inputmaps
        z = convn(inputMap{map}, ones(poollayer.scale) / (poollayer.scale ^ 2), 'valid');   %用kron卷积实现平均池化
        downSample{map,1} = z(1 : poollayer.scale : end, 1 : poollayer.scale : end, :); %保存下采样结果
        poolMap{map,1} = downSample{map,1} + poollayer.b{map};  %根据采样步长跳读取值,加偏置,无激活函数
        if poollayer.weight %若池化层有权值
            poolMap{map,1} = poollayer.w{map} .* downSample{map,1} + poollayer.b{map};  %根据采样步长跳读取值,加权重,无激活函数
        end
        if poollayer.function %若池化层有sigmoid传递函数
            poolMap{map,1} = sigmoid(poollayer.w{map} .* downSample{map,1} + poollayer.b{map});  %根据采样步长跳读取值,计算激活函数
        end
        maxPos{map,1} = 1/(poollayer.scale ^ 2) .* ones(height,width,datanum); %平均池化每个元素的概率都是1/(poollayer.scale^2)
    end
end
end