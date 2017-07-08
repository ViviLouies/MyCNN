 function convMap = cnnConvolve(inputMap,convlayer)
%inputMap ����ͼcell��ʽ��cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%convlayer �����(����)
%convMap ���������������cell��ʽ
inputmaps = numel(inputMap); %����inputmaps��Ŀ
datanum = size(inputMap{1},3);  %����inputmaps��С
outputsize = convlayer.mapsize ;  %outputmaps�ĳߴ�
convMap = cell(convlayer.outputmaps,1);   %Ԥ�ȿ���outputmaps�Ĵ洢�ռ�
for i = 1 : convlayer.outputmaps   %����˵���Ŀ����outputmaps��Ŀ
    convtemp = zeros(outputsize(1),outputsize(2),datanum);  %��ʱ����������һ��inputmap�����Ľ��outputmaps
    for j = 1:inputmaps
        z = convn(inputMap{j,1},rot180(convlayer.w{i,j}),'valid'); %һ����������ξ��ÿһ��inputmaps(������貽��Ϊ1)
        %convn�������Զ���ת�����,����ҪԤ����תһ�� 
        convtemp = convtemp + z(1:convlayer.stride:end,1:convlayer.stride:end,:); %���ݲ�������,�����(Ȩֵ�������)
    end
    switch convlayer.function
        case 'sigmoid'
            convMap{i,1} = sigmoid(convtemp + convlayer.b{i,1}); %��ƫ�ã�����sigmoid���
        case 'tanh'
            convMap{i,1} = tanh(convtemp + convlayer.b{i,1}); %��ƫ�ã�����tanh���
        case 'relu'
            convMap{i,1} = relu(convtemp + convlayer.b{i,1}); %��ƫ�ã�����relu���
        otherwise
            error('Unknown function of convolutional layer!')
    end
end