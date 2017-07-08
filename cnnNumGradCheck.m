function cnnNumGradCheck(net,x,y)
epsilion = 1e-4;  %扰动
error_limit = 1e-6; %精度
layer_num = numel(net.layers); %层数
format long;
%% 检查每层的权值和偏置
for layer = layer_num: -1 : 2
    switch net.layers{layer}.type
        case'conv' %卷积层
            for i = 1:net.layers{layer}.outputmaps
                net_rp = net; net_lm = net;
                net_rp.layers{layer}.b{i,1} = net_rp.layers{layer}.b{i,1} + epsilion;
                net_lm.layers{layer}.b{i,1} = net_lm.layers{layer}.b{i,1} - epsilion;
                net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                e = abs(d - net.layers{layer}.db{i});
                %disp(d);
                %disp(net.layers{layer}.db{i});
                fprintf('conv-%d-b-%0.9f\n',layer,e);
                if e > error_limit
                    error('numerical gradient checking failed!')
                end
                for j = 1:net.layers{layer-1}.outputmaps
                    for u = 1 : size(net.layers{layer}.w{i,j}, 1)
                        for v = 1 : size(net.layers{layer}.w{i,j}, 2)
                            net_rp = net; net_lm = net;
                            net_rp.layers{layer}.w{i,j}(u,v) = net_rp.layers{layer}.w{i,j}(u,v) + epsilion;
                            net_lm.layers{layer}.w{i,j}(u,v) = net_lm.layers{layer}.w{i,j}(u,v) - epsilion;
                            net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                            net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                            d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                            e = abs(d - net.layers{layer}.dw{i,j}(u,v));
                            %disp(d);
                            %disp(net.layers{layer}.dw{i,j}(u,v));
                            fprintf('conv-%d-w-%0.9f\n',layer,e);
                            if e > error_limit
                                error('numerical gradient checking failed!')
                            end
                        end
                    end
                end
            end
        case  'pool' %池化层
            for i = 1:net.layers{layer}.outputmaps
                net_rp = net; net_lm = net;
                net_rp.layers{layer}.b{i,1} = net_rp.layers{layer}.b{i,1} + epsilion;
                net_lm.layers{layer}.b{i,1} = net_lm.layers{layer}.b{i,1} - epsilion;
                net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                e = abs(d - net.layers{layer}.db{i});
                fprintf('pool-%d-b-%0.9f\n',layer,e);
                if e > error_limit
                    error('numerical gradient checking failed!')
                end
            end
            if net.layers{layer}.weight || net.layers{layer}.function %如果有权值或传递函数
                for i = 1:net.layers{layer}.outputmaps
                    net_rp = net; net_lm = net;
                    net_rp.layers{layer}.w{i} = net_rp.layers{layer}.w{i,1} + epsilion;
                    net_lm.layers{layer}.w{i} = net_lm.layers{layer}.w{i,1} - epsilion;
                    net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                    net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                    d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                    e = abs(d - net.layers{layer}.dw{i});
                    fprintf('pool-%d-w-%0.9f\n',layer,e);
                    if e > error_limit
                        error('numerical gradient checking failed!')
                    end
                end
            end
        case 'bn'  %batch normalization层
            for i = 1:net.layers{layer}.outputmaps
                net_rp = net; net_lm = net;
                net_rp.layers{layer}.gamma{i,1} = net_rp.layers{layer}.gamma{i,1} + epsilion;
                net_lm.layers{layer}.gamma{i,1} = net_lm.layers{layer}.gamma{i,1} - epsilion;
                net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                e = abs(d - net.layers{layer}.dgamma{i});
                %disp(d);
                %disp(net.layers{layer}.db{i});
                fprintf('bn-%d-gamma-%0.9f\n',layer,e);
                if e > error_limit
                    error('numerical gradient checking failed!')
                end
                net_rp = net; net_lm = net;
                net_rp.layers{layer}.beta{i,1} = net_rp.layers{layer}.beta{i,1} + epsilion;
                net_lm.layers{layer}.beta{i,1} = net_lm.layers{layer}.beta{i,1} - epsilion;
                net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                e = abs(d - net.layers{layer}.dbeta{i});
                %disp(d);
                %disp(net.layers{layer}.db{i});
                fprintf('bn-%d-beta-%0.9f\n',layer,e);
                if e > error_limit
                    error('numerical gradient checking failed!')
                end
            end
        case 'fc'
            % 检查全连接层 net.layers{layer_num}.b
            for i = 1:numel(net.layers{layer}.b)
                net_rp = net; net_lm = net;
                net_rp.layers{layer}.b(i) = net_rp.layers{layer}.b(i) + epsilion;
                net_lm.layers{layer}.b(i) = net_lm.layers{layer}.b(i) - epsilion;
                net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                e = abs(d - net.layers{layer}.db(i));
                fprintf('fc-%d-b-%0.9f\n',layer,e);
                if e > error_limit
                    error('numerical gradient checking failed!')
                end
            end
            % 检查全连接层 net.layers{layer_num}.w
            for i = 1:size(net.layers{layer}.w,1)
                for j = 1:size(net.layers{layer}.w,2)
                    net_rp = net; net_lm = net;
                    net_rp.layers{layer}.w(i,j) = net_rp.layers{layer}.w(i,j) + epsilion;
                    net_lm.layers{layer}.w(i,j) = net_lm.layers{layer}.w(i,j) - epsilion;
                    net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                    net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                    d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                    e = abs(d - net.layers{layer}.dw(i,j));
                    fprintf('fc-%d-w-%0.9f\n',layer,e);
                    if e > error_limit
                        error('numerical gradient checking failed!')
                    end
                end
            end
        case 'loss'
            % 检查全连接层 net.layers{layer_num}.b
            for i = 1:numel(net.layers{layer}.b)
                net_rp = net; net_lm = net;
                net_rp.layers{layer}.b(i) = net_rp.layers{layer}.b(i) + epsilion;
                net_lm.layers{layer}.b(i) = net_lm.layers{layer}.b(i) - epsilion;
                net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                e = abs(d - net.layers{layer}.db(i));
                fprintf('loss-%d-b-%0.9f\n',layer,e);
                if e > error_limit
                    error('numerical gradient checking failed!')
                end
            end
            % 检查全连接层 net.layers{layer_num}.w
            for i = 1:size(net.layers{layer}.w,1)
                for j = 1:size(net.layers{layer}.w,2)
                    net_rp = net; net_lm = net;
                    net_rp.layers{layer}.w(i,j) = net_rp.layers{layer}.w(i,j) + epsilion;
                    net_lm.layers{layer}.w(i,j) = net_lm.layers{layer}.w(i,j) - epsilion;
                    net_lm = cnnFF(net_lm, x); net_lm = cnnBP(net_lm, y);
                    net_rp = cnnFF(net_rp, x); net_rp = cnnBP(net_rp, y);
                    d = (net_rp.loss - net_lm.loss) / (2 * epsilion);
                    e = abs(d - net.layers{layer}.dw(i,j));
                    fprintf('loss-%d-w-%0.9f\n',layer,e);
                    if e > error_limit
                        error('numerical gradient checking failed!')
                    end
                end
            end
        otherwise
            error('Unknow layer''s type!');
    end
end
format short;
end