function net = cnnbp_addPrune(net)

global fid

    n = numel(net.layers);
    
% %     %   error
% %     net.e = net.o - y;
% %     %  loss function
% %     net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas

% NEW - sparsity error
    sparsityError = 0;
    
    featureVectorRequired = 0;
    for l = n : -1 : 1
        if strcmp(net.layers{l}.type, 'f')
            if l == n
                net.layers{l}.d = net.e .* net.d_act_fun(net.o);
            else
% NEW - nonSparsityPenalty for internal/hidden FC layers
                if (net.nonSparsityPenalty>0)
                    pi = repmat(net.layers{l}.p, size(net.layers{l}.a, 2), 1);
                    sparsityError = [zeros(size(net.layers{l}.a, 2), 1) net.nonSparsityPenalty ...
                        * (-net.sparsityTarget ./ pi + (1 - net.sparsityTarget) ./(1-pi))];
                end

                net.layers{l}.d = (net.layers{l+1}.W' * net.layers{l+1}.d + sparsityError).* net.d_act_fun(net.layers{l}.a);
                
% NEW - dropOutMask for deltas of internal/hidden FC layers                
                if (net.dropoutFraction>0)
                    net.layers{l}.d = net.layers{l}.d .* [net.layers{l}.dropOutMask];
                end 
            end
            
            if not(strcmp(net.layers{l-1}.type, 'f'))
                featureVectorRequired = 1;
            end

            
        else
            if featureVectorRequired == 1
                featureVectorDelta = (net.layers{l+1}.W' * net.layers{l+1}.d);
                sa = size(net.layers{l}.a{1});
                fan_outSize = sa(1) * sa(2);
                for j = 1:numel(net.layers{l}.a)
                    net.layers{l}.d{j} = reshape(featureVectorDelta(((j-1) * fan_outSize + 1) : j*fan_outSize, :), sa(1), sa(2), sa(3));
                end
                featureVectorRequired = 0;

% % %     net.od = net.e .* double(net.o>0); % (net.o .* (1 - net.o));   %  output delta   
% % %     net.fvd = (net.ffW' * net.od);              %  feature vector delta
% % %     if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
% % %         net.fvd = net.fvd .* (net.fv .* (1 - net.fv)); % double(net.fv>0); % 
% % %     end
% % % 
% % %     %  reshape feature vector deltas into output map style
% % %     sa = size(net.layers{n}.a{1});
% % %     fvnum = sa(1) * sa(2);
% % %     for j = 1 : numel(net.layers{n}.a)
% % %         net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
% % %     end

            else
                if strcmp(net.layers{l}.type, 'c')
                    for j = 1 : numel(net.layers{l}.a)  
                        if net.layers{l}.used_maps(j)           
                            net.layers{l}.d{j} = net.d_act_fun(net.layers{l}.a{j})...
                                .* (expand(net.layers{l + 1}.d{j}, ...
                                    [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) ...
                                    / net.layers{l + 1}.scale ^ 2);
                        else
                             net.layers{l}.d{j} = net.d_act_fun(net.layers{l}.a{j}) * 0;
                        end
                    end
                    
                elseif strcmp(net.layers{l}.type, 's')
                    for i = 1 : numel(net.layers{l}.a)
                        z = zeros(size(net.layers{l}.a{1}));
                        for j = 1 : numel(net.layers{l + 1}.a)
                             z = z + convn(net.layers{l + 1}.d{j}, flip(flip(net.layers{l + 1}.k{i}{j}, 1), 2), 'full');
                        end
                        net.layers{l}.d{i} = z;
                    end
                    
                end
            end
        end
    end
    
% % %     function X = rot180(X)
% % %         X = flipdim(flipdim(X, 1), 2);
% % %     end
    
    %%  calc gradients
    firstFClayer = 1;
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
            
        
        elseif strcmp(net.layers{l}.type, 'f')
            if firstFClayer == 1
                net.layers{l}.dW = net.layers{l}.d * net.featureVector'/ size(net.layers{l}.d, 2);
                net.layers{l}.db = zeros(net.layers{l}.size, 1);
                
                firstFClayer = 0;
            else
                net.layers{l}.dW = net.layers{l}.d * (net.layers{l-1}.a)' / size(net.layers{l}.d, 2);
                net.layers{l}.db = zeros(net.layers{l}.size, 1);
            end
        end
    end

    
% % %     net.dffW = net.od * (net.fv)' / size(net.od, 2);
% % %     net.dffb = 0;%mean(net.od, 2);


end