% ѵ���õ�CNN�������
function [acc, ind] = cnnTest(net, data, label)
    %%CNNǰ�����
    net = cnnFF(net, data);
    outlayer = numel(net.layers);
    [~, a] = max(net.layers{outlayer}.a); %�ҵ��������ֵ��Ӧ�����
    [~, y] = max(label); %�ҵ����ݵ���ʵ���
    ind = find(y ~= a);

    err = numel(ind) / size(label, 2); %���������
    acc = 1.0 - err;
    fprintf('Testing accuracy = %f%%.\n',acc*100);
end