function net = cnnWeightUpdate(net, opts)
%opts.alpha为学习率
for layer = 2 : numel(net.layers)
    switch net.layers{layer}.type
        case  'conv' %卷积层权值更新
            for outputmap = 1 : net.layers{layer}.outputmaps
                for inputmap = 1: net.layers{layer-1}.outputmaps
                    % 权值更新的公式：W_new = W_old - alpha * de/dW（SGD误差对权值导数）
                    net.layers{layer}.mw{outputmap,inputmap} = opts.momentum * net.layers{layer}.mw{outputmap,inputmap} - opts.alpha * net.layers{layer}.dw{outputmap,inputmap}; %计算动量项
                    net.layers{layer}.w{outputmap,inputmap} = net.layers{layer}.w{outputmap,inputmap} + net.layers{layer}.mw{outputmap,inputmap}; %权值更新
                    %net.layers{layer}.w{outputmap,inputmap} = net.layers{layer}.w{outputmap,inputmap} - opts.alpha * net.layers{layer}.dw{outputmap,inputmap};
                end
                % 偏置更新的公式：b_new = b_old - alpha * de/db（SGD误差对权值导数）
                net.layers{layer}.mb{outputmap} = opts.momentum * net.layers{layer}.mb{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
                net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} + net.layers{layer}.mb{outputmap};
                %net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
            end
        case  'pool'%池化层权值更新
            for outputmap = 1 : net.layers{layer}.outputmaps
                % 偏置更新的公式：b_new = b_old - alpha * de/db（SGD误差对权值导数）
                net.layers{layer}.mb{outputmap} = opts.momentum * net.layers{layer}.mb{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
                net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} + net.layers{layer}.mb{outputmap};
                %net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
                if net.layers{layer}.weight || net.layers{layer}.function %若池化层有权值或传递函数(默认有权值)
                    % 权值更新的公式：W_new = W_old - alpha * de/dW（SGD误差对权值导数）
                    net.layers{layer}.mw{outputmap} = opts.momentum * net.layers{layer}.mw{outputmap} - opts.alpha * net.layers{layer}.dw{outputmap};
                    net.layers{layer}.w{outputmap} = net.layers{layer}.w{outputmap} + net.layers{layer}.mw{outputmap};
                    %net.layers{layer}.w{outputmap} = net.layers{layer}.w{outputmap} - opts.alpha * net.layers{layer}.dw{outputmap};
                end
            end
        case  'bn' %batch normalization层权值更新
            for outputmap = 1:net.layers{layer}.outputmaps
                net.layers{layer}.mgamma{outputmap,1} = opts.momentum * net.layers{layer}.mgamma{outputmap,1} - opts.alpha * net.layers{layer}.dgamma{outputmap,1};%计算动量项
                net.layers{layer}.gamma{outputmap,1} = net.layers{layer}.gamma{outputmap,1} + net.layers{layer}.mgamma{outputmap,1}; %gamma更新
                net.layers{layer}.mbeta{outputmap,1} = opts.momentum * net.layers{layer}.mbeta{outputmap,1} - opts.alpha * net.layers{layer}.dbeta{outputmap,1};%计算动量项
                net.layers{layer}.beta{outputmap,1} = net.layers{layer}.beta{outputmap,1} + net.layers{layer}.mbeta{outputmap,1}; %beta更新
            end
        case 'fc' %全连接层权值更新
            % 权值更新的公式：W_new = W_old - alpha * de/dW（SGD误差对权值导数）
            net.layers{layer}.mw = opts.momentum * net.layers{layer}.mw - opts.alpha * net.layers{layer}.dw;
            net.layers{layer}.w = net.layers{layer}.w + net.layers{layer}.mw ;
            %net.layers{layer}.w = net.layers{layer}.w - opts.alpha * net.layers{layer}.dw;
            % 偏置更新的公式：b_new = b_old - alpha * de/db（SGD误差对权值导数）
            net.layers{layer}.mb = opts.momentum * net.layers{layer}.mb - opts.alpha * net.layers{layer}.db;
            net.layers{layer}.b = net.layers{layer}.b + net.layers{layer}.mb;
            %net.layers{layer}.b = net.layers{layer}.b - opts.alpha * net.layers{layer}.db;
        case 'loss' %输出层权值更新
            % 权值更新的公式：W_new = W_old - alpha * de/dW（SGD误差对权值导数）
            net.layers{layer}.mw = opts.momentum * net.layers{layer}.mw - opts.alpha * net.layers{layer}.dw;
            net.layers{layer}.w = net.layers{layer}.w + net.layers{layer}.mw ;
            %net.layers{layer}.w = net.layers{layer}.w - opts.alpha * net.layers{layer}.dw;
            % 偏置更新的公式：b_new = b_old - alpha * de/db（SGD误差对权值导数）
            net.layers{layer}.mb = opts.momentum * net.layers{layer}.mb - opts.alpha * net.layers{layer}.db;
            net.layers{layer}.b = net.layers{layer}.b + net.layers{layer}.mb;
            %net.layers{layer}.b = net.layers{layer}.b - opts.alpha * net.layers{layer}.db;
    end
end
end