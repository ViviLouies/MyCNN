function bnlayer = batch_normlization(inputMap,bnlayer)
%inputMap ����ͼcell��ʽ��cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%bnlayer batch normalization��(����)
%normMap batch normalization���������cell��ʽ
batch_num = size(inputMap{1},3);  %����mini-batch��Ŀ(��ÿһ��ͼ����Ŀ)
for i = 1:bnlayer.outputmaps
    bnlayer.mean{i,1} = mean(inputMap{i,1},3); %����ͼ�ľ�ֵ
    bnlayer.z_decent{i,1} = inputMap{i,1} - repmat(bnlayer.mean{i,1},[1,1,batch_num]); %����ͼ��ÿ��sliceȥ���Ļ�
    bnlayer.var{i,1} = mean((bnlayer.z_decent{i,1}).^2,3);  %����ͼ�ķ����ƫ���ƣ�
    bnlayer.std{i,1} = sqrt(bnlayer.var{i,1}+bnlayer.epsilion); %����ͼ�ı�׼���ƫ��epsilionƽ����
    bnlayer.z_norm{i,1} = bnlayer.z_decent{i,1} ./ repmat(bnlayer.std{i,1},[1,1,batch_num]); % z-score��׼��
    bnlayer.a{i,1} = bnlayer.gamma{i} .* bnlayer.z_norm{i,1} + bnlayer.beta{i}; %ӳ���ع�
end
end