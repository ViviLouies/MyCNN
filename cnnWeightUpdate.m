function net = cnnWeightUpdate(net, opts)
%opts.alphaΪѧϰ��
for layer = 2 : numel(net.layers)
    switch net.layers{layer}.type
        case  'conv' %�����Ȩֵ����
            for outputmap = 1 : net.layers{layer}.outputmaps
                for inputmap = 1: net.layers{layer-1}.outputmaps
                    % Ȩֵ���µĹ�ʽ��W_new = W_old - alpha * de/dW��SGD����Ȩֵ������
                    net.layers{layer}.mw{outputmap,inputmap} = opts.momentum * net.layers{layer}.mw{outputmap,inputmap} - opts.alpha * net.layers{layer}.dw{outputmap,inputmap}; %���㶯����
                    net.layers{layer}.w{outputmap,inputmap} = net.layers{layer}.w{outputmap,inputmap} + net.layers{layer}.mw{outputmap,inputmap}; %Ȩֵ����
                    %net.layers{layer}.w{outputmap,inputmap} = net.layers{layer}.w{outputmap,inputmap} - opts.alpha * net.layers{layer}.dw{outputmap,inputmap};
                end
                % ƫ�ø��µĹ�ʽ��b_new = b_old - alpha * de/db��SGD����Ȩֵ������
                net.layers{layer}.mb{outputmap} = opts.momentum * net.layers{layer}.mb{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
                net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} + net.layers{layer}.mb{outputmap};
                %net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
            end
        case  'pool'%�ػ���Ȩֵ����
            for outputmap = 1 : net.layers{layer}.outputmaps
                % ƫ�ø��µĹ�ʽ��b_new = b_old - alpha * de/db��SGD����Ȩֵ������
                net.layers{layer}.mb{outputmap} = opts.momentum * net.layers{layer}.mb{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
                net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} + net.layers{layer}.mb{outputmap};
                %net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
                if net.layers{layer}.weight || net.layers{layer}.function %���ػ�����Ȩֵ�򴫵ݺ���(Ĭ����Ȩֵ)
                    % Ȩֵ���µĹ�ʽ��W_new = W_old - alpha * de/dW��SGD����Ȩֵ������
                    net.layers{layer}.mw{outputmap} = opts.momentum * net.layers{layer}.mw{outputmap} - opts.alpha * net.layers{layer}.dw{outputmap};
                    net.layers{layer}.w{outputmap} = net.layers{layer}.w{outputmap} + net.layers{layer}.mw{outputmap};
                    %net.layers{layer}.w{outputmap} = net.layers{layer}.w{outputmap} - opts.alpha * net.layers{layer}.dw{outputmap};
                end
            end
        case  'bn' %batch normalization��Ȩֵ����
            for outputmap = 1:net.layers{layer}.outputmaps
                net.layers{layer}.mgamma{outputmap,1} = opts.momentum * net.layers{layer}.mgamma{outputmap,1} - opts.alpha * net.layers{layer}.dgamma{outputmap,1};%���㶯����
                net.layers{layer}.gamma{outputmap,1} = net.layers{layer}.gamma{outputmap,1} + net.layers{layer}.mgamma{outputmap,1}; %gamma����
                net.layers{layer}.mbeta{outputmap,1} = opts.momentum * net.layers{layer}.mbeta{outputmap,1} - opts.alpha * net.layers{layer}.dbeta{outputmap,1};%���㶯����
                net.layers{layer}.beta{outputmap,1} = net.layers{layer}.beta{outputmap,1} + net.layers{layer}.mbeta{outputmap,1}; %beta����
            end
        case 'fc' %ȫ���Ӳ�Ȩֵ����
            % Ȩֵ���µĹ�ʽ��W_new = W_old - alpha * de/dW��SGD����Ȩֵ������
            net.layers{layer}.mw = opts.momentum * net.layers{layer}.mw - opts.alpha * net.layers{layer}.dw;
            net.layers{layer}.w = net.layers{layer}.w + net.layers{layer}.mw ;
            %net.layers{layer}.w = net.layers{layer}.w - opts.alpha * net.layers{layer}.dw;
            % ƫ�ø��µĹ�ʽ��b_new = b_old - alpha * de/db��SGD����Ȩֵ������
            net.layers{layer}.mb = opts.momentum * net.layers{layer}.mb - opts.alpha * net.layers{layer}.db;
            net.layers{layer}.b = net.layers{layer}.b + net.layers{layer}.mb;
            %net.layers{layer}.b = net.layers{layer}.b - opts.alpha * net.layers{layer}.db;
        case 'loss' %�����Ȩֵ����
            % Ȩֵ���µĹ�ʽ��W_new = W_old - alpha * de/dW��SGD����Ȩֵ������
            net.layers{layer}.mw = opts.momentum * net.layers{layer}.mw - opts.alpha * net.layers{layer}.dw;
            net.layers{layer}.w = net.layers{layer}.w + net.layers{layer}.mw ;
            %net.layers{layer}.w = net.layers{layer}.w - opts.alpha * net.layers{layer}.dw;
            % ƫ�ø��µĹ�ʽ��b_new = b_old - alpha * de/db��SGD����Ȩֵ������
            net.layers{layer}.mb = opts.momentum * net.layers{layer}.mb - opts.alpha * net.layers{layer}.db;
            net.layers{layer}.b = net.layers{layer}.b + net.layers{layer}.mb;
            %net.layers{layer}.b = net.layers{layer}.b - opts.alpha * net.layers{layer}.db;
    end
end
end