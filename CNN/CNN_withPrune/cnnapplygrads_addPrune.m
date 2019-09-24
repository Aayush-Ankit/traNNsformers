function net = cnnapplygrads_addPrune(net, opts)

global fid

    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                end
                if(opts.learn_bias)
                    net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
                end
            end
        
        elseif strcmp(net.layers{l}.type, 'f')
% NEW - weightPenaltyL2
           if (net.weightPenaltyL2>0)
               dW = net.layers{l}.dW + net.weightPenaltyL2 * [zeros(size(net.layers{l}.W, 1), 1) net.layers{l}.W(:, 2:end)];
           else
               dW = net.layers{l}.dW;
           end
           
           dW = opts.alpha * dW;
           
           if (opts.momentum>0)
               net.layers{l}.vW = opts.momentum * net.layers{l}.vW + dW;
               dW = net.layers{l}.vW;
           end
           
           net.layers{l}.W = net.layers{l}.W - dW;
           net.layers{l}.b = net.layers{l}.b - opts.alpha * net.layers{l}.b;
        
        end
    end
    

% % %     net.ffW = net.ffW - opts.alpha * net.dffW;
% % %     net.ffb = net.ffb - opts.alpha * net.dffb;
end