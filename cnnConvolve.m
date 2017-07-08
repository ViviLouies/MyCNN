 function convMap = cnnConvolve(inputMap,convlayer)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%convlayer 卷积层(参数)
%convMap 卷积输出结果，还是cell格式
inputmaps = numel(inputMap); %读入inputmaps数目
datanum = size(inputMap{1},3);  %读入inputmaps大小
outputsize = convlayer.mapsize ;  %outputmaps的尺寸
convMap = cell(convlayer.outputmaps,1);   %预先开辟outputmaps的存储空间
for i = 1 : convlayer.outputmaps   %卷积核的数目，即outputmaps数目
    convtemp = zeros(outputsize(1),outputsize(2),datanum);  %临时变量，保存一个inputmap卷积后的结果outputmaps
    for j = 1:inputmaps
        z = convn(inputMap{j,1},rot180(convlayer.w{i,j}),'valid'); %一个卷积核依次卷积每一个inputmaps(这里假设步长为1)
        %convn函数会自动旋转卷积核,所以要预先旋转一下 
        convtemp = convtemp + z(1:convlayer.stride:end,1:convlayer.stride:end,:); %根据步长采样,并求和(权值共享策略)
    end
    switch convlayer.function
        case 'sigmoid'
            convMap{i,1} = sigmoid(convtemp + convlayer.b{i,1}); %加偏置，计算sigmoid结果
        case 'tanh'
            convMap{i,1} = tanh(convtemp + convlayer.b{i,1}); %加偏置，计算tanh结果
        case 'relu'
            convMap{i,1} = relu(convtemp + convlayer.b{i,1}); %加偏置，计算relu结果
        otherwise
            error('Unknown function of convolutional layer!')
    end
end