function net = cnnBP(net,label)
%% ����в�����ȣ�
lambda = 1e-4; %softmax���Ȩ��˥��ϵ��
layer_num = numel(net.layers); %�������
for layer = layer_num : -1 : 1
    switch net.layers{layer}.type
        %=======================%�������ʧ�㣬���ȼ���loss���ٷ��򴫲�%===================%
        case 'loss'
            net.layers{layer}.error = net.layers{layer}.a - label; %ʵ��������������֮������
            if strcmp(net.layers{layer}.function, 'sigmoid')
                net.loss =  1/2* sum(net.layers{layer}.error(:) .^ 2) / size(net.layers{layer}.a, 2);  %���ۺ��������þ���������Ϊ���ۺ���
                net.layers{layer}.delta = net.layers{layer}.error .* (net.layers{layer}.a .* (1 - net.layers{layer}.a)); %�����в�sigmoid���ݺ���
            elseif strcmp(net.layers{layer}.function, 'tanh')
                net.loss =  1/2* sum(net.layers{layer}.error(:) .^ 2) / size(net.layers{layer}.a, 2);  %���ۺ��������þ���������Ϊ���ۺ���
                net.layers{layer}.delta = net.layers{layer}.error .* (1 - (net.layers{layer}.a).^2); %�����в�tanh���ݺ���
            elseif strcmp(net.layers{layer}.function, 'softmax')
                net.loss = -1/size(net.layers{layer}.a, 2) * label(:)' * log(net.layers{layer}.a(:)) + lambda/2 * sum(net.layers{layer}.w(:) .^ 2);  %softmax��ʧ����������Ȩ��˥�������������
                net.layers{layer}.delta = net.layers{layer}.a - label;  %softmax���������
            end
            %����ʧ���ǰһ���Ǿ���㡢�ػ����batch normalization�㣬���������һ���դ�㣨ʸ�����㣬���������ػ���outputmaps��������������Ҫ�������������������
            if strcmp(net.layers{layer-1}.type, 'pool') %��ȫ���Ӳ��ǰһ���ǳػ���(Ĭ�ϳػ����޼����)
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %������դ���������
                if net.layers{layer-1}.function   %���ػ�����sigmoid���ݺ���������Ըò�����ƫ����������ǰȫ���Ӳ�����룩
                    net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                end
                %��ʸ����������դ��������ȣ�����������չ����ǰһ��ػ��������map��ͬ�ĳߴ���ʽ
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %ȡǰһ������map�ߴ�
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %ǰһ�������map�ĸ���
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            elseif strcmp(net.layers{layer-1}.type, 'bn')  %���ȫ���Ӳ��ǰһ����batch normalization�㣨�޼������,ֱ�ӷ�ʸ����������
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %������դ���������
                %��ʸ����������դ��������ȣ�����������չ����ǰһ��ػ��������map��ͬ�ĳߴ���ʽ
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %ȡǰһ������map�ߴ�
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %ǰһ�������map�ĸ���
                    z = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                    net.layers{layer-1}.dvar{outputmap,1} = -1/2 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z .* net.layers{layer-1}.z_decent{outputmap,1},3) ./ (net.layers{layer-1}.std{outputmap,1}).^3; %�����
                    net.layers{layer-1}.dmean{outputmap,1} = -1 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z,3) ./ net.layers{layer-1}.std{outputmap,1}...
                        - 2 .* net.layers{layer-1}.dvar{outputmap,1} .* mean(net.layers{layer-1}.z_decent{outputmap,1},3);     %��ֵ����
                    net.layers{layer-1}.delta{outputmap,1} = net.layers{layer-1}.gamma{outputmap,1} .* z ./ repmat(net.layers{layer-1}.std{outputmap,1},[1,1,data_num])...
                        + 2/data_num .* repmat(net.layers{layer-1}.dvar{outputmap,1},[1,1,data_num]) .* net.layers{layer-1}.z_decent{outputmap,1}...
                        + 1/data_num .* repmat(net.layers{layer-1}.dmean{outputmap,1},[1,1,data_num]);     %�в�
                    net.layers{layer-1}.dgamma{outputmap,1} = sum(sum(sum(z .* net.layers{layer-1}.z_norm{outputmap,1}))) ./ data_num;  %gammaƫ����
                    net.layers{layer-1}.dbeta{outputmap,1} = sum(z(:)) ./ data_num;  %betaƫ����
                end
            elseif strcmp(net.layers{layer-1}.type, 'conv')  %���ȫ���Ӳ��ǰһ���Ǿ���㣨�м������,��в�Ҫ����ǰһ������ƫ����
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta; %������դ���������
                switch net.layers{layer-1}.function
                    case 'sigmoid'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                    case 'tanh'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (1 - (net.layers{layer}.input).^2);
                    case 'relu'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* double(net.layers{layer}.input>0.0);
                end
                %��ʸ����������դ��������ȣ�����������չ����ǰһ�����������map��ͬ�ĳߴ���ʽ
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %ȡǰһ������map�ߴ�
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %ǰһ�������map�ĸ���
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            end
            %=======================%�����ȫ���Ӳ㣬�����д�ͳ��BP�㷨%===================%
        case 'fc'
            %���㵱ǰȫ���Ӳ��������
            if strcmp(net.layers{layer}.function, 'sigmoid')
            net.layers{layer}.delta = (net.layers{layer+1}.w' * net.layers{layer+1}.delta) .* (net.layers{layer}.a .* (1 - net.layers{layer}.a)); 
            elseif strcmp(net.layers{layer}.function, 'tanh')
                net.layers{layer}.delta = (net.layers{layer+1}.w' * net.layers{layer+1}.delta) .* (1 - (net.layers{layer}.a).^2);
            elseif strcmp(net.layers{layer}.function, 'relu')
                net.layers{layer}.delta = (net.layers{layer+1}.w' * net.layers{layer+1}.delta) .* double(net.layers{layer}.a>0.0);
            end
            %����ȫ���Ӳ��ǰһ���Ǿ���㡢�ػ����batch normalization�㣬���������һ���դ�㣨ʸ�����㣬���������ػ���outputmaps��������������Ҫ�������������������
            if strcmp(net.layers{layer-1}.type, 'pool') %��ȫ���Ӳ��ǰһ���ǳػ���(Ĭ�ϳػ����޼����)
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %�������������
                if net.layers{layer-1}.function   %���ػ�����sigmoid���ݺ���������Ըò�����ƫ����������ǰȫ���Ӳ�����룩
                    net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                end
                %��ʸ����������դ��������ȣ�����������չ����ǰһ��ػ��������map��ͬ�ĳߴ���ʽ
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %ȡǰһ������map�ߴ�
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %ǰһ�������map�ĸ���
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            elseif strcmp(net.layers{layer-1}.type, 'bn')  %���ȫ���Ӳ��ǰһ����batch normalization�㣨�޼������,ֱ�ӷ�ʸ����������
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta;  %������դ���������
                %��ʸ����������դ��������ȣ�����������չ����ǰһ��ػ��������map��ͬ�ĳߴ���ʽ
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %ȡǰһ������map�ߴ�
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %ǰһ�������map�ĸ���
                    z = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                    net.layers{layer-1}.dvar{outputmap,1} = -1/2 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z .* net.layers{layer-1}.z_decent{outputmap,1},3) ./ (net.layers{layer-1}.std{outputmap,1}).^3; %�����
                    net.layers{layer-1}.dmean{outputmap,1} = -1 .* net.layers{layer-1}.gamma{outputmap,1} .* sum(z,3) ./ net.layers{layer-1}.std{outputmap,1}...
                        - 2 .* net.layers{layer-1}.dvar{outputmap,1} .* mean(net.layers{layer-1}.z_decent{outputmap,1},3);     %��ֵ����
                    net.layers{layer-1}.delta{outputmap,1} = net.layers{layer-1}.gamma{outputmap,1} .* z ./ repmat(net.layers{layer-1}.std{outputmap,1},[1,1,data_num])...
                        + 2/data_num .* repmat(net.layers{layer-1}.dvar{outputmap,1},[1,1,data_num]) .* net.layers{layer-1}.z_decent{outputmap,1}...
                        + 1/data_num .* repmat(net.layers{layer-1}.dmean{outputmap,1},[1,1,data_num]);     %�в�
                    net.layers{layer-1}.dgamma{outputmap,1} = sum(sum(sum(z .* net.layers{layer-1}.z_norm{outputmap,1}))) ./ data_num;  %gammaƫ����
                    net.layers{layer-1}.dbeta{outputmap,1} = sum(z(:)) ./ data_num;  %betaƫ����
                end
            elseif strcmp(net.layers{layer-1}.type, 'conv')  %���ȫ���Ӳ��ǰһ���Ǿ���㣨�м������,��в�Ҫ����ǰһ������ƫ����
                net.layers{layer}.raster_delta = net.layers{layer}.w' * net.layers{layer}.delta; %�������������
                switch net.layers{layer-1}.function
                    case 'sigmoid'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (net.layers{layer}.input .* (1 - net.layers{layer}.input));
                    case 'tanh'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* (1 - (net.layers{layer}.input).^2);
                    case 'relu'
                        net.layers{layer}.raster_delta = net.layers{layer}.raster_delta .* double(net.layers{layer}.input>0.0);
                end
                %��ʸ����������դ��������ȣ�����������չ����ǰһ�����������map��ͬ�ĳߴ���ʽ
                [height, width, data_num] = size(net.layers{layer-1}.a{1}); %ȡǰһ������map�ߴ�
                maparea = height * width;
                for outputmap = 1 : numel(net.layers{layer-1}.a)  %ǰһ�������map�ĸ���
                    net.layers{layer-1}.delta{outputmap,1} = reshape(net.layers{layer}.raster_delta((outputmap - 1) * maparea + 1: outputmap * maparea, :), height, width, data_num);
                end
            end
            %====%����Ǿ���㣬������ϲ���(����maxPosλ�ñ����ͳһ���ػ���ƽ���ػ��������ȼ���)%====%
        case 'conv' %��������ֻ�����Ǿ���㡢�ػ����batch normalization�㣬һ�㲻��ֱ����ȫ���Ӳ����ʧ�㣨����������������������ͣ���������Ҳ�ڼ����դ���ʱ�������ˣ�
            if strcmp(net.layers{layer+1}.type, 'pool') %�����������ǳػ���
                for outputmap = 1 : net.layers{layer}.outputmaps  %�ò�����map�ĸ���
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
                    if net.layers{layer+1}.weight || net.layers{layer+1}.function %������ĳػ�����Ȩ�ػ򴫵ݺ���(Ĭ�ϰ���Ȩ��),����ݲв�ݺ�����Ҫ���ϳػ����Ȩ��
                        net.layers{layer}.delta{outputmap,1} = net.layers{layer}.delta{outputmap,1} * net.layers{layer+1}.w{outputmap};
                    end
                end
            elseif strcmp(net.layers{layer+1}.type, 'bn') %������������batch normalization��(��pool�����ƴ���)
                for outputmap = 1 : net.layers{layer}.outputmaps  %�ò�����map�ĸ���
                    switch net.layers{layer}.function
                        case 'sigmoid'
                            net.layers{layer}.delta{outputmap,1} = net.layers{layer}.a{outputmap} .* (1 - net.layers{layer}.a{outputmap}) .* net.layers{layer+1}.delta{outputmap,1};      
                        case 'tanh'
                            net.layers{layer}.delta{outputmap,1} = (1 - (net.layers{layer}.a{outputmap}).^2) .* net.layers{layer+1}.delta{outputmap,1};
                        case 'relu'
                            net.layers{layer}.delta{outputmap,1} = double(net.layers{layer}.a{outputmap}>0.0) .* net.layers{layer+1}.delta{outputmap,1}; 
                    end
                end
            elseif strcmp(net.layers{layer+1}.type, 'conv') %�����������Ǿ����
                for inputmap = 1 : net.layers{layer}.outputmaps %��layer������map�ĸ���
                    z = zeros(size(net.layers{layer}.a{1}));
                    for outputmap = 1 : net.layers{layer+1}.outputmaps %��layer+1������map�ĸ���
                        %���ⲽ��ʱ������
                        padMap = map_padding(net.layers{layer+1}.delta{outputmap,1},net.layers{layer+1}.mapsize,net.layers{layer+1}.kernelsize,net.layers{layer+1}.stride);
                        switch net.layers{layer}.function
                            case 'sigmoid'
                                z = z + net.layers{layer}.a{inputmap} .* (1 - net.layers{layer}.a{inputmap})...
                                    .* convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid'); %����convn���Զ���ת����ˣ������ﲻ����ת
                            case 'tanh'
                                z = z + (1 - (net.layers{layer}.a{inputmap}).^2)...
                                    .* convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid'); %����convn���Զ���ת����ˣ������ﲻ����ת
                            case 'relu'
                                z = z + double(net.layers{layer}.a{inputmap}>0.0)...
                                    .* convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid'); %����convn���Զ���ת����ˣ������ﲻ����ת)
                        end
                    end
                    net.layers{layer}.delta{inputmap,1} = z;
                end
            end
            %=======================%����ǳػ��㣬������²���%===================%
        case 'pool'
            if strcmp(net.layers{layer+1}.type, 'fc') || strcmp(net.layers{layer+1}.type, 'loss')
                continue;  %Ĭ�ϳػ����һ���Ǿ����,����ȫ���Ӳ����ʧ�㣬���������ڹ�դ���Ѽ��㣬�˴�����ʡ��
            else
                for inputmap = 1 : net.layers{layer}.outputmaps %��layer������map�ĸ���,�͵�layer-1����������map��Ŀ��ͬ
                    z = zeros(size(net.layers{layer}.a{1}));
                    for outputmap = 1 : net.layers{layer+1}.outputmaps %��layer+1������map�ĸ���
                        %���ⲽ��ʱ������
                        padMap = map_padding(net.layers{layer+1}.delta{outputmap,1},net.layers{layer+1}.mapsize,net.layers{layer+1}.kernelsize,net.layers{layer+1}.stride);
                        z = z + convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid');
                        %����convn���Զ���ת����ˣ������ﲻ����ת
                    end
                    net.layers{layer}.delta{inputmap,1} = z;
                    if net.layers{layer}.function %���ػ�����sigmoid���ݺ���
                        net.layers{layer}.delta{inputmap,1} = z .* net.layers{layer}.a{inputmap} .* (1 - net.layers{layer}.a{inputmap});
                    end
                end
            end
        %=======================%�����batch normalization��%===================%
        case 'bn' %batch normalization��һ���Ӿ���㣬�Ҳ��ͳػ�����������
            if strcmp(net.layers{layer+1}.type, 'fc') || strcmp(net.layers{layer+1}.type, 'loss')
                continue;  %Ĭ��batch normalization���һ���Ǿ����,����ȫ���Ӳ����ʧ�㣬���������ڹ�դ���Ѽ��㣬�˴�����ʡ��
            elseif strcmp(net.layers{layer+1}.type, 'conv') %(��pool�����ƴ���)
                 for inputmap = 1 : net.layers{layer}.outputmaps %��layer������map�ĸ���,�͵�layer-1����������map��Ŀ��ͬ
                    z = zeros(size(net.layers{layer}.a{1}));
                    for outputmap = 1 : net.layers{layer+1}.outputmaps %��layer+1������map�ĸ���
                        %���ⲽ��ʱ������
                        padMap = map_padding(net.layers{layer+1}.delta{outputmap,1},net.layers{layer+1}.mapsize,net.layers{layer+1}.kernelsize,net.layers{layer+1}.stride);
                        z = z + convn(padMap,(net.layers{layer+1}.w{outputmap,inputmap}), 'valid');
                        %����convn���Զ���ת����ˣ������ﲻ����ת
                    end
                    data_num = size(net.layers{layer}.a{1},3); %mini-batch ��Ŀ
                    net.layers{layer}.dvar{inputmap,1} = -1/2 .* net.layers{layer}.gamma{inputmap,1} .* sum(z .* net.layers{layer}.z_decent{inputmap,1},3) ./ (net.layers{layer}.std{inputmap,1}).^3; %�����
                    net.layers{layer}.dmean{inputmap,1} = -1 .* net.layers{layer}.gamma{inputmap,1} .* sum(z,3) ./ net.layers{layer}.std{inputmap,1}...     
                        - 2 .* net.layers{layer}.dvar{inputmap,1} .* mean(net.layers{layer}.z_decent{inputmap,1},3);     %��ֵ����
                    net.layers{layer}.delta{inputmap,1} = net.layers{layer}.gamma{inputmap,1} .* z ./ repmat(net.layers{layer}.std{inputmap,1},[1,1,data_num])...
                        + 2/data_num .* repmat(net.layers{layer}.dvar{inputmap,1},[1,1,data_num]) .* net.layers{layer}.z_decent{inputmap,1}...
                        + 1/data_num .* repmat(net.layers{layer}.dmean{inputmap,1},[1,1,data_num]);     %�в�
                    net.layers{layer}.dgamma{inputmap,1} = sum(sum(sum(z .* net.layers{layer}.z_norm{inputmap,1}))) ./ data_num;  %gammaƫ����
                    net.layers{layer}.dbeta{inputmap,1} = sum(z(:)) ./ data_num;  %betaƫ����
                 end
            else
                error('situation undefined: %s + %s','bn',net.layers{layer+1}.type);
            end
    end
end

%% �����ݶ�
for layer = 2:layer_num
    switch net.layers{layer}.type
        %========================%��������ĵ���%===========================%
        case  'conv'
            for outputmap = 1:net.layers{layer}.outputmaps
                for inputmap = 1:net.layers{layer-1}.outputmaps
                    padMap = map_padding(net.layers{layer}.delta{outputmap},net.layers{layer}.mapsize,[1,1],net.layers{layer}.stride);
                    %���Ǿ����Ĳ�����Ҫ�������������ȸ��ݾ�����������ڲ����䣨��ʵ��һ�������ڰ����������ϲ�������0��䣻���ڲ��ý����ⲿ��䣬���Խ�kernelsize����Ϊ[1,1]��
                    z = convn(net.layers{layer-1}.a{inputmap},flipall(padMap), 'valid');
                    %convn���Զ���ת�����,����Ҫ����ת����,flipall�����������ÿһά��(����������ά��)����ת��180��
                    net.layers{layer}.dw{outputmap,inputmap} = z./size(net.layers{layer}.delta{outputmap}, 3);
                end
                net.layers{layer}.db{outputmap,1} = sum(net.layers{layer}.delta{outputmap}(:)) / size(net.layers{layer}.delta{outputmap}, 3);
            end
            %=======================%����ػ���ĵ���%============================%
        case 'pool'
            for outputmap = 1:net.layers{layer}.outputmaps
                net.layers{layer}.db{outputmap,1} = sum(net.layers{layer}.delta{outputmap}(:)) / size(net.layers{layer}.delta{outputmap}, 3);
                if net.layers{layer}.weight || net.layers{layer}.function %���ػ�����Ȩֵ�򴫵ݺ���(Ĭ����Ȩֵ)
                    net.layers{layer}.dw{outputmap,1} = sum(net.layers{layer}.delta{outputmap}(:) .* net.layers{layer}.downsample{outputmap}(:)) / size(net.layers{layer}.delta{outputmap}, 3);
                end
            end
            %==================%����ȫ���Ӳ�ĵ���%====================%
        case 'fc'
            net.layers{layer}.dw = net.layers{layer}.delta * (net.layers{layer}.input)' / size(net.layers{layer}.delta,2);
            net.layers{layer}.db = mean(net.layers{layer}.delta, 2);
            %==================%������ʧ��ĵ���%====================%
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