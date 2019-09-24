function net = cnnff_addPrune(net, x, y)

global fid


    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    unused_inputs = rand(size(x)) < net.first_layer_dropout;
    net.layers{1}.a{1}(unused_inputs) = 0;
    inputmaps = 1;
    
    firstFClayer = 1;
    
    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                if net.layers{l}.used_maps(j)
                    %  create temp output map
                    z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                    for i = 1 : inputmaps   %  for each input map
                        %  convolve with corresponding kernel and add to temp output map
                        z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                    end
                    %  add bias, pass through nonlinearity
                    net.layers{l}.a{j} = net.act_fun(z + net.layers{l}.b{j});
                else
                    net.layers{l}.a{j} = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                end
            end

            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
            
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
            
        elseif strcmp(net.layers{l}.type, 'f')
            if firstFClayer == 1
                %  concatenate all end layer feature maps into vector
                net.featureVector = [];
                for j = 1 : numel(net.layers{l-1}.a)
                    sa = size(net.layers{l-1}.a{j});
                    net.featureVector = [net.featureVector; reshape(net.layers{l-1}.a{j}, sa(1) * sa(2), sa(3))];
                end
                %  feedforward into output perceptrons
                net.layers{l}.a = max(0, net.layers{l}.W * net.featureVector);% + repmat(net.ffb, 1, size(net.fv, 2)));
                
                firstFClayer = 0;             
                
            elseif firstFClayer == 0
                net.layers{l}.a = max(0, net.layers{l}.W * net.layers{l-1}.a);
            end
            
            if (l ~= n)
% NEW - dropOutMask for activations of the internal/hidden FC layers (i.e. FC
% layers, except the last one)
                if(net.dropoutFraction>0)
                    if(net.testing)
                        net.layers{l}.a = (net.layers{l}.a) .*(1-net.dropoutFraction);
                    else
                        net.layers{l}.dropOutMask = (rand(size(net.layers{l}.a))>net.dropoutFraction);
                        net.layers{l}.a = net.layers{l}.a.*net.layers{l}.dropOutMask;
                    end
                end
% NEW - nonSparsityPenalty for internal/hidden FC layer
                if(net.nonSparsityPenalty>0)
                    net.layers{l}.p = 0.99 * net.layers{l}.p + 0.01 * mean(net.layers{l}.a, 2);
                end
            end
        end
    end
    
    net.o = net.layers{n}.a;
    
    
    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
    
   

% % %     %  concatenate all end layer feature maps into vector
% % %     net.fv = [];
% % %     for j = 1 : numel(net.layers{n}.a)
% % %         sa = size(net.layers{n}.a{j});
% % %         net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
% % %     end
% % %     %  feedforward into output perceptrons
% % %     net.o = max(0, net.ffW * net.fv);% + repmat(net.ffb, 1, size(net.fv, 2)));

end