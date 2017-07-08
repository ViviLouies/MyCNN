% %{ 网络定义：
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
%net_def 网络定义
%inputSize 输入图像的尺寸[m,n]
%outputSize 标签(one of c形式)的尺寸，即c，也就是输出层神经元的个数，分多少个类，自然就有多少个输出神经元
net = net_def;
for layer = 1 : numel(net.layers)   % 对每一层进行判断并操作
    switch net.layers{layer}.type
        case 'input' %输入层
            net.layers{layer}.outputmaps = 1;   %输入层的featuremap特征图就1个，即原始图像
            net.layers{layer}.mapsize = inputSize; %输入层的每个特征图的尺寸
        case 'conv' %卷积层
            if ~isfield(net_def.layers{layer},'stride')%如果未定义步长，默认为1
                net.layers{layer}.stride = 1;
            end
            if ~isfield(net_def.layers{layer},'function')%如果未定义步长，默认为1
                net.layers{layer}.function = 'sigmoid';
            end
            pre_judge = net.layers{layer-1}.mapsize - net.layers{layer}.kernelsize;
            if sum(pre_judge) == 0 && net.layers{layer}.stride > 1
                error('the convolutional outputmap just has only one element => the stride should less than 1');
            end
            net.layers{layer}.mapsize = (net.layers{layer-1}.mapsize - net.layers{layer}.kernelsize)/net.layers{layer}.stride + 1; %更新卷积层的outputmaps的尺寸
            assert(all(floor(net.layers{layer}.mapsize)==net.layers{layer}.mapsize), ['Layer ' num2str(layer) ' size must be an integer. Actual: ' num2str(net.layers{layer}.mapsize)]);%不能整除时，报错，需更换卷积核尺寸
            kernelarea = prod(net.layers{layer}.kernelsize); %卷积核的面积，prod计算数组的连乘积, eg. prod([1,2,3]) = 1*2*3 = 6;
            fan_out = net.layers{layer}.outputmaps * kernelarea;  %连接到后一层卷积核的权值W参数个数(不考虑共享)
            fan_in = net.layers{layer-1}.outputmaps * kernelarea;  %连接到前一层卷积核的权值W参数个数(不考虑共享)
            for i = 1 : net.layers{layer}.outputmaps  %对于卷积层的每一个outputmap(等于卷积核的个数)
                for j = 1: net.layers{layer-1}.outputmaps %对于卷积层每一个inputmaps(等于前一层的outputmaps)
                    net.layers{layer}.w{i,j} = (rand(net.layers{layer}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out)); %初始化卷积层权值(Xavier方法)
                    net.layers{layer}.mw{i,j} = zeros(net.layers{layer}.kernelsize);  %权值更新的动量项（权值），初始化为0
                end
                net.layers{layer}.b{i,1} = 0;  %初始化卷积核偏置为零,每个outputmap一个bias,并非每个卷积核一个bias
                net.layers{layer}.mb{i,1}=0;  %权值更新的动量项（偏置），初始化为0
            end
        case 'pool' %池化层
            if ~isfield(net_def.layers{layer},'stride')%如果未定义步长，默认为1
                net.layers{layer}.stride = 1;
            end
            net.layers{layer}.outputmaps = net.layers{layer-1}.outputmaps;  %池化层的特征图个数outputmaps和前一层一致
            net.layers{layer}.mapsize = net.layers{layer-1}.mapsize./net.layers{layer}.scale;   %更新池化层的outputmaps的尺寸
            assert(all(floor(net.layers{layer}.mapsize)==net.layers{layer}.mapsize), ['Layer ' num2str(layer) ' size must be integer. Actual: ' num2str(net.layers{layer}.mapsize)]);%不能整除时，报错，需更换步长
            %注：池化层可以有以下四种情况：(1)没有参数(2)有偏置(本程序默认)(3)有权值(默认包含偏置)(4)有传递函数(默认包含权值和偏置)
            for i = 1 : net.layers{layer}.outputmaps   %对于层内的每个特征图
                net.layers{layer}.b{i,1} = 0;  %初始化池化层的偏置为零(若池化层只有偏置，没有权值=>mean pooling权值相当于1/4*[1,1;1,1],max pooling权值相当于[0,1;0,0],1的位置由最大值确定）
                net.layers{layer}.mb{i,1} = 0;  %权值更新的动量项（偏置），初始化为0
                if net.layers{layer}.weight || net.layers{layer}.function %若池化层有权值或传递函数（默认有权值）
                    net.layers{layer}.w{i,1} = 1;  %初始化池化层的权重为1
                    net.layers{layer}.mw{i,1} = 0; %权值更新的动量项（权值），初始化为0
                end
            end
        case 'bn' %batch normalization层
            net.layers{layer}.outputmaps = net.layers{layer-1}.outputmaps;  %batch normalization层的特征图个数outputmaps和前一层一致
            net.layers{layer}.mapsize = net.layers{layer-1}.mapsize;   %batch normalization层的outputmaps的尺寸和上一层一致
            for i =1:net.layers{layer}.outputmaps  %对于层内的每个特征图（都有一对gamma和beta）
                net.layers{layer}.gamma{i,1} = 1;   %初始化映射重构权值gamma为1
                net.layers{layer}.mgamma{i,1} = 0;  %初始化映射重构权值项的动量项为0
                net.layers{layer}.beta{i,1} = 0;    %初始化映射重构偏置beta为0
                net.layers{layer}.mbeta{i,1} = 0;   %初始化映射重构权值的动量项为0
                net.layers{layer}.epsilion = 1e-10; %标准差平滑项
            end
        case 'fc'  %全连接层
            if ~isfield(net_def.layers{layer},'function')%如果未定义步长，默认为1
                net.layers{layer}.function = 'sigmoid';
            end
            fcnum = prod(net.layers{layer-1}.mapsize) * net.layers{layer-1}.outputmaps;
            %fcnum 是前面一层的神经元个数,这一层的上一层是池化层，包含有net_def.layers{layer-1}.outputmaps个特征map,每个特征map的大小是net_def.layers{layer-1}.mapsize
            %所以，该层的神经元个数是 特征map数目 * （每个特征map的大小,高和宽->若全连接层的前一层是卷积层或池化层，则长和宽可能大于1，若是全连接层，则长和宽均为1）
            net.layers{layer}.mapsize = [1,1];  %全连接层每个outputmap神经元的尺寸均为1*1
            net.layers{layer}.w= (rand(net.layers{layer}.outputmaps, fcnum) - 0.5) * 2 * sqrt(6 / (net.layers{layer}.outputmaps + fcnum));   %初始化全连接层权值(Xavier方法)
            net.layers{layer}.mw = zeros(net.layers{layer}.outputmaps, fcnum);  %权值更新的动量项（权值），初始化为0
            net.layers{layer}.b= zeros(net.layers{layer}.outputmaps, 1);  %初始化全连接层偏置为0
            net.layers{layer}.mb = zeros(net.layers{layer}.outputmaps, 1);  %权值更新的动量项（偏置），初始化为0
        case 'loss'  %损失层,即最后一层（一般也是全连接层）
            if ~isfield(net_def.layers{layer},'function')%如果未定义步长，默认为1
                net.layers{layer}.function = 'sigmoid';
            end
            fcnum = prod(net.layers{layer-1}.mapsize) * net.layers{layer-1}.outputmaps; %输出层的前面一层的神经元个数
            net.layers{layer}.w= (rand(outputSize, fcnum) - 0.5) * 2 * sqrt(6 / (outputSize + fcnum));   %初始化损失层权值(Xavier方法)
            net.layers{layer}.mw = zeros(outputSize, fcnum);  %权值更新的动量项（权值），初始化为0
            net.layers{layer}.b= zeros(outputSize, 1);  %初始化损失层偏置为0
            net.layers{layer}.mb = zeros(outputSize, 1);  %权值更新的动量项（偏置），初始化为0
        otherwise
            error('Unknow layer''s type!');
    end
end