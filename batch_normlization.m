function bnlayer = batch_normlization(inputMap,bnlayer)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%bnlayer batch normalization层(参数)
%normMap batch normalization结果，还是cell格式
batch_num = size(inputMap{1},3);  %读入mini-batch数目(即每一批图像数目)
for i = 1:bnlayer.outputmaps
    bnlayer.mean{i,1} = mean(inputMap{i,1},3); %特征图的均值
    bnlayer.z_decent{i,1} = inputMap{i,1} - repmat(bnlayer.mean{i,1},[1,1,batch_num]); %特征图的每个slice去中心化
    bnlayer.var{i,1} = mean((bnlayer.z_decent{i,1}).^2,3);  %特征图的方差（有偏估计）
    bnlayer.std{i,1} = sqrt(bnlayer.var{i,1}+bnlayer.epsilion); %特征图的标准差（有偏，epsilion平滑）
    bnlayer.z_norm{i,1} = bnlayer.z_decent{i,1} ./ repmat(bnlayer.std{i,1},[1,1,batch_num]); % z-score标准化
    bnlayer.a{i,1} = bnlayer.gamma{i} .* bnlayer.z_norm{i,1} + bnlayer.beta{i}; %映射重构
end
end