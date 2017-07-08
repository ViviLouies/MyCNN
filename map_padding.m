function padMap = map_padding(delta,mapSize,kernelSize,stride)
%delta��Ҫ����map
%mapSize��delta��ͼ�γߴ�
%kernelSize������˵ĳߴ�
%stride������Ĳ���
%padMap�������䣨0�����map
padMap = delta; %��padMap��ʼ��Ϊdelta��������������Ļ���
pad_out = kernelSize - 1; %delta�ⲿ���ߴ�
pad_in = stride - 1;  %delta�ڲ����ߴ磨���ڲ�Ԫ�ؼ����С��
mapsize = mapSize + (mapSize-1)*pad_in;
%delta�ڲ������ĳߴ� = deltaԭ�ߴ� + ԭ��Ԫ�صļ����mapSize-1��* Ԫ��֮�����0����Ŀpad_in
%padSize = pad_out*2 + mapsize;
%delta�����ĳߴ� = �ⲿ���ߴ�*2 + �ڲ����ߴ�
datanum = size(delta,3);
%% ������ڲ�(�����Ҫ)
if pad_in
    map  = zeros([mapsize, datanum]);%��ʼ���ڲ�����ľ���
    for i = 1:mapSize(1)
        for j = 1:mapSize(2)
            map(i+(i-1)*pad_in, j+(j-1)*pad_in,:)= delta(i,j,:); %���ڲ����ߴ������
        end
    end
    padMap = map;
end
%% ������ⲿ(�����Ҫ)
if sum(pad_out)
    padMap = padarray(padMap,[pad_out,0]);  %��padarray��������ⲿ
end
end