% %{ ���綨�壺
% cnn.layers = {
%     struct('type', 'input') %input layer
%     struct('type', 'conv', 'outputmaps', 24, 'kernelsize', [5,5],'stride', 2) %convolution layer
%     struct('type', 'pool', 'scale', 2) %subsampling layer
%     struct('type', 'conv', 'outputmaps', 72, 'kernelsize', [3,3]) %convolution layer
%     struct('type', 'pool', 'scale', 2) %subsampling layer
%     struct('type', 'conv', 'outputmaps', 144, 'kernelsize',[3,3]) %convolution layer
%     struct('type', 'pool', 'scale', 2) %subsampling layer
%     struct('type', 'fc') %full connecting layer
%     };
% %}

function net = cnnSetup(net_def, inputSize, outputSize)
%net_def ���綨��
%inputSize ����ͼ��ĳߴ�[m,n]
%outputSize ��ǩ(one of c��ʽ)�ĳߴ磬��c��Ҳ�����������Ԫ�ĸ������ֶ��ٸ��࣬��Ȼ���ж��ٸ������Ԫ
net = net_def;
for layer = 1 : numel(net.layers)   % ��ÿһ������жϲ�����
    switch net.layers{layer}.type
        case 'input' %�����
            net.layers{layer}.outputmaps = 1;   %������featuremap����ͼ��1������ԭʼͼ��
            net.layers{layer}.mapsize = inputSize; %������ÿ������ͼ�ĳߴ�
        case 'conv' %�����
            if ~isfield(net_def.layers{layer},'stride')%���δ���岽����Ĭ��Ϊ1
                net.layers{layer}.stride = 1;
            end
            if ~isfield(net_def.layers{layer},'function')%���δ���岽����Ĭ��Ϊ1
                net.layers{layer}.function = 'sigmoid';
            end
            pre_judge = net.layers{layer-1}.mapsize - net.layers{layer}.kernelsize;
            if sum(pre_judge) == 0 && net.layers{layer}.stride > 1
                error('the convolutional outputmap just has only one element => the stride should less than 1');
            end
            net.layers{layer}.mapsize = (net.layers{layer-1}.mapsize - net.layers{layer}.kernelsize)/net.layers{layer}.stride + 1; %���¾�����outputmaps�ĳߴ�
            assert(all(floor(net.layers{layer}.mapsize)==net.layers{layer}.mapsize), ['Layer ' num2str(layer) ' size must be an integer. Actual: ' num2str(net.layers{layer}.mapsize)]);%��������ʱ���������������˳ߴ�
            kernelarea = prod(net.layers{layer}.kernelsize); %����˵������prod������������˻�, eg. prod([1,2,3]) = 1*2*3 = 6;
            fan_out = net.layers{layer}.outputmaps * kernelarea;  %���ӵ���һ�����˵�ȨֵW��������(�����ǹ���)
            fan_in = net.layers{layer-1}.outputmaps * kernelarea;  %���ӵ�ǰһ�����˵�ȨֵW��������(�����ǹ���)
            for i = 1 : net.layers{layer}.outputmaps  %���ھ�����ÿһ��outputmap(���ھ���˵ĸ���)
                for j = 1: net.layers{layer-1}.outputmaps %���ھ����ÿһ��inputmaps(����ǰһ���outputmaps)
                    net.layers{layer}.w{i,j} = (rand(net.layers{layer}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out)); %��ʼ�������Ȩֵ(Xavier����)
                    net.layers{layer}.mw{i,j} = zeros(net.layers{layer}.kernelsize);  %Ȩֵ���µĶ����Ȩֵ������ʼ��Ϊ0
                end
                net.layers{layer}.b{i,1} = 0;  %��ʼ�������ƫ��Ϊ��,ÿ��outputmapһ��bias,����ÿ�������һ��bias
                net.layers{layer}.mb{i,1}=0;  %Ȩֵ���µĶ����ƫ�ã�����ʼ��Ϊ0
            end
        case 'pool' %�ػ���
            if ~isfield(net_def.layers{layer},'stride')%���δ���岽����Ĭ��Ϊ1
                net.layers{layer}.stride = 1;
            end
            net.layers{layer}.outputmaps = net.layers{layer-1}.outputmaps;  %�ػ��������ͼ����outputmaps��ǰһ��һ��
            net.layers{layer}.mapsize = net.layers{layer-1}.mapsize./net.layers{layer}.scale;   %���³ػ����outputmaps�ĳߴ�
            assert(all(floor(net.layers{layer}.mapsize)==net.layers{layer}.mapsize), ['Layer ' num2str(layer) ' size must be integer. Actual: ' num2str(net.layers{layer}.mapsize)]);%��������ʱ���������������
            %ע���ػ���������������������(1)û�в���(2)��ƫ��(������Ĭ��)(3)��Ȩֵ(Ĭ�ϰ���ƫ��)(4)�д��ݺ���(Ĭ�ϰ���Ȩֵ��ƫ��)
            for i = 1 : net.layers{layer}.outputmaps   %���ڲ��ڵ�ÿ������ͼ
                net.layers{layer}.b{i,1} = 0;  %��ʼ���ػ����ƫ��Ϊ��(���ػ���ֻ��ƫ�ã�û��Ȩֵ=>mean poolingȨֵ�൱��1/4*[1,1;1,1],max poolingȨֵ�൱��[0,1;0,0],1��λ�������ֵȷ����
                net.layers{layer}.mb{i,1} = 0;  %Ȩֵ���µĶ����ƫ�ã�����ʼ��Ϊ0
                if net.layers{layer}.weight || net.layers{layer}.function %���ػ�����Ȩֵ�򴫵ݺ�����Ĭ����Ȩֵ��
                    net.layers{layer}.w{i,1} = 1;  %��ʼ���ػ����Ȩ��Ϊ1
                    net.layers{layer}.mw{i,1} = 0; %Ȩֵ���µĶ����Ȩֵ������ʼ��Ϊ0
                end
            end
        case 'bn' %batch normalization��
            net.layers{layer}.outputmaps = net.layers{layer-1}.outputmaps;  %batch normalization�������ͼ����outputmaps��ǰһ��һ��
            net.layers{layer}.mapsize = net.layers{layer-1}.mapsize;   %batch normalization���outputmaps�ĳߴ����һ��һ��
            for i =1:net.layers{layer}.outputmaps  %���ڲ��ڵ�ÿ������ͼ������һ��gamma��beta��
                net.layers{layer}.gamma{i,1} = 1;   %��ʼ��ӳ���ع�ȨֵgammaΪ1
                net.layers{layer}.mgamma{i,1} = 0;  %��ʼ��ӳ���ع�Ȩֵ��Ķ�����Ϊ0
                net.layers{layer}.beta{i,1} = 0;    %��ʼ��ӳ���ع�ƫ��betaΪ0
                net.layers{layer}.mbeta{i,1} = 0;   %��ʼ��ӳ���ع�Ȩֵ�Ķ�����Ϊ0
                net.layers{layer}.epsilion = 1e-10; %��׼��ƽ����
            end
        case 'fc'  %ȫ���Ӳ�
            if ~isfield(net_def.layers{layer},'function')%���δ���岽����Ĭ��Ϊ1
                net.layers{layer}.function = 'sigmoid';
            end
            fcnum = prod(net.layers{layer-1}.mapsize) * net.layers{layer-1}.outputmaps;
            %fcnum ��ǰ��һ�����Ԫ����,��һ�����һ���ǳػ��㣬������net_def.layers{layer-1}.outputmaps������map,ÿ������map�Ĵ�С��net_def.layers{layer-1}.mapsize
            %���ԣ��ò����Ԫ������ ����map��Ŀ * ��ÿ������map�Ĵ�С,�ߺͿ�->��ȫ���Ӳ��ǰһ���Ǿ�����ػ��㣬�򳤺Ϳ���ܴ���1������ȫ���Ӳ㣬�򳤺Ϳ��Ϊ1��
            net.layers{layer}.mapsize = [1,1];  %ȫ���Ӳ�ÿ��outputmap��Ԫ�ĳߴ��Ϊ1*1
            net.layers{layer}.w= (rand(net.layers{layer}.outputmaps, fcnum) - 0.5) * 2 * sqrt(6 / (net.layers{layer}.outputmaps + fcnum));   %��ʼ��ȫ���Ӳ�Ȩֵ(Xavier����)
            net.layers{layer}.mw = zeros(net.layers{layer}.outputmaps, fcnum);  %Ȩֵ���µĶ����Ȩֵ������ʼ��Ϊ0
            net.layers{layer}.b= zeros(net.layers{layer}.outputmaps, 1);  %��ʼ��ȫ���Ӳ�ƫ��Ϊ0
            net.layers{layer}.mb = zeros(net.layers{layer}.outputmaps, 1);  %Ȩֵ���µĶ����ƫ�ã�����ʼ��Ϊ0
        case 'loss'  %��ʧ��,�����һ�㣨һ��Ҳ��ȫ���Ӳ㣩
            if ~isfield(net_def.layers{layer},'function')%���δ���岽����Ĭ��Ϊ1
                net.layers{layer}.function = 'sigmoid';
            end
            fcnum = prod(net.layers{layer-1}.mapsize) * net.layers{layer-1}.outputmaps; %������ǰ��һ�����Ԫ����
            net.layers{layer}.w= (rand(outputSize, fcnum) - 0.5) * 2 * sqrt(6 / (outputSize + fcnum));   %��ʼ����ʧ��Ȩֵ(Xavier����)
            net.layers{layer}.mw = zeros(outputSize, fcnum);  %Ȩֵ���µĶ����Ȩֵ������ʼ��Ϊ0
            net.layers{layer}.b= zeros(outputSize, 1);  %��ʼ����ʧ��ƫ��Ϊ0
            net.layers{layer}.mb = zeros(outputSize, 1);  %Ȩֵ���µĶ����ƫ�ã�����ʼ��Ϊ0
        otherwise
            error('Unknow layer''s type!');
    end
end