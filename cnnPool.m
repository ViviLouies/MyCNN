function [downSample,poolMap,maxPos] = cnnPool(inputMap,poollayer)
%inputMap ����ͼcell��ʽ��cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%poollayer �ػ������
%downSample �����²��������δ�������ݺ������ɿ��ɳػ������룩,�����²����㷴�򴫲������е��ݶȼ���
%poolMap �ػ����������cell��ʽ
%maxPos ���ֵλ�ü�¼���������򴫲�
inputmaps = numel(inputMap);  %����inputmaps��Ŀ����outputmaps��Ŀ
[height,width,datanum] = size(inputMap{1});  %����inputmaps��С
stride = poollayer.scale;   %����Ĭ��2
downSample = cell(inputmaps,1); %%Ԥ�ȿ��ٿռ䣬�������²�����������ڳػ�����ȨֵʱȨֵ�ĸ��£�
poolMap = cell(inputmaps,1); %Ԥ�ȿ��ٿռ䣬����ػ�������²��� + ���㣩
maxPos = cell(inputmaps,1); %Ԥ�ȿ��ٿռ䣬�������ֵλ��
%����maxPos��Ƿ������ͳһ��������������ȼ���
if strcmp(poollayer.method, 'max') %��������ػ�
    for map = 1:inputmaps
        downSample{map,1} = zeros(height/poollayer.scale,width/poollayer.scale,datanum); %��ʼ���²�������Ϊ0
        poolMap{map,1} = zeros(height/poollayer.scale,width/poollayer.scale,datanum); %��ʼ���ػ�����Ϊ0
        maxPos{map,1} = zeros(height,width,datanum); %��ʼ�����ֵλ�þ���Ϊ0
        for row = 1:stride:height
            for col = 1:stride:width
                patch = inputMap{map}(row:row+stride-1,col:col+stride-1,:); %stride*stride patch
                [val,ind] = max(reshape(patch,[stride^2,datanum]));  % �ҳ����ֵ����λ��
                downSample{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = val; %�����²������
                poolMap{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = val + poollayer.b{map};  % ��ƫ��,max pooling,�޼����
                if poollayer.weight %���ػ�����Ȩֵ
                    poolMap{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = poollayer.w{map} .* val + poollayer.b{map};  % ��Ȩ�غ�ƫ��,max pooling,�޼����
                end
                if poollayer.function %���ػ�����sigmoid���ݺ���
                    poolMap{map,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = sigmoid(poollayer.w{map} .* val + poollayer.b{map});  % ��Ȩ�غ�ƫ��,max pooling,���㼤���
                end
                ind_row = rem(ind,stride); %�ҵ����ֵ������Ӧ��������(��stride^2��λ��)
                ind_row(ind_row==0) = stride; %stride�ı���ȡ���Ϊ0��Ӧ�ӻ�ȥ
                ind_col = round(ind/stride); %�ҵ����ֵ������Ӧ��������
                for i = 1:datanum
                    maxPos{map,1}(row + ind_row(i) - 1, col + ind_col(i) - 1, i) = 1; %�Ƴ����ֵλ����ԭͼ�е���Ӧλ�ã���Ϊ1
                end
            end
        end
    end
elseif strcmp(poollayer.method, 'mean') %�����ƽ���ػ�
    for map = 1:inputmaps
        z = convn(inputMap{map}, ones(poollayer.scale) / (poollayer.scale ^ 2), 'valid');   %��kron���ʵ��ƽ���ػ�
        downSample{map,1} = z(1 : poollayer.scale : end, 1 : poollayer.scale : end, :); %�����²������
        poolMap{map,1} = downSample{map,1} + poollayer.b{map};  %���ݲ�����������ȡֵ,��ƫ��,�޼����
        if poollayer.weight %���ػ�����Ȩֵ
            poolMap{map,1} = poollayer.w{map} .* downSample{map,1} + poollayer.b{map};  %���ݲ�����������ȡֵ,��Ȩ��,�޼����
        end
        if poollayer.function %���ػ�����sigmoid���ݺ���
            poolMap{map,1} = sigmoid(poollayer.w{map} .* downSample{map,1} + poollayer.b{map});  %���ݲ�����������ȡֵ,���㼤���
        end
        maxPos{map,1} = 1/(poollayer.scale ^ 2) .* ones(height,width,datanum); %ƽ���ػ�ÿ��Ԫ�صĸ��ʶ���1/(poollayer.scale^2)
    end
end
end