function net = cnnsetup_addPrune(net, x, y)

% Initialize first layer (input image sample)
% inputmaps = # of input channels
inputmaps = 1; 
% mapsize = dimensions of the input space
mapsize = size(squeeze(x(:, :, 1)));

net.n = numel(net.layers);

% Setting up each layer within CNN
for l = 1 : numel(net.layers) 
    
    % POOLING layer settings
    if strcmp(net.layers{l}.type, 's')
        mapsize = mapsize / net.layers{l}.scale;
        % Check if scalling ratio is acceptable (should be integer
        % scalling ratio)
        assert(all(floor(mapsize)==mapsize), ...
            ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
        for j = 1 : inputmaps
            net.layers{l}.b{j} = 0; % Setting variable for bias
        end
    end

    % CONVOLUTIONAL layer settings
    if strcmp(net.layers{l}.type, 'c') 
        % n_{h,w}_new = n_{h,w} - f + 1 (assumming no padding and stride=1)
        mapsize = mapsize - net.layers{l}.kernelsize + 1;
        % Number of parameters (weights/nodes) on the output/on this layer
        fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
        
        %  outputmaps = # of filters/kernels on this layer
        for j = 1 : net.layers{l}.outputmaps
            % Number of parameters (weights/nodes) on the input/on the prior layer
            fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
            for i = 1 : inputmaps
                % Setting up parameters (these will be changed during training)
                net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
            end
            % Setting up biases (these will be changed during training)
            net.layers{l}.b{j} = 0;
        end
        % Change the # of input channels to the # of filters/kernels used
        % in this layer
        inputmaps = net.layers{l}.outputmaps;
    end
    
    % FULLY CONNECTED layer settings
    if strcmp(net.layers{l}.type, 'f')
        fan_in = prod(mapsize) * inputmaps;
        fan_out = net.layers{l}.size;
        net.layers{l}.W = (rand(fan_out, fan_in)-0.5) * 0.01 * 2;        
%         net.layers{l}.W = (rand(fan_out, fan_in)-0.5) * sqrt(6 / (fan_out + fan_in));
        net.layers{l}.vW = zeros(size(net.layers{l}.W));
        net.layers{l}.b = zeros(fan_out, 1);
        mapsize = net.layers{l}.size;
        inputmaps = 1;

% NEW - average activations (for use with sparsity)
        net.layers{l}.p = zeros(1,net.layers{l}.size);
        
        
        % TRANNSFORMER
        % new addition - prune threshold & prune map
        net.pruneth{l} = 0;
        net.pmap{l} = ones(size(net.layers{l}.W));
        % new addition - cluster prune factor
        net.cluster_prune_factor{l} = 0.005;
        % new addition - cluster threshold & cluster map
        net.cmap{l} = ones(size(net.layers{l}.W));
        % new addition - overall map 
        net.map{l} = ones(size(net.layers{l}.W));
        
        % initialize layer-wise prune-curr & prune_prev
        net.unclustered_prev{l} = 0;
        net.unclustered_curr{l} = 0;
        % track cluster_count for debug
        net.cluster_count{l} = 0;
               

    end
    
end

% % % % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
% % % % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
% % % % 'ffb' is the biases of the output neurons.
% % % % 'ffW' is the weights between the last layer and the output neurons. 
% % % %  Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
% % % fvnum = prod(mapsize) * inputmaps;
% % % onum = size(y, 1);
% % % if onum==1
% % %     warning('Output layer consists of only 1 neuron!')
% % % end
% % % 
% % % net.ffb = zeros(onum, 1);
% % % %net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
% % % net.ffW = (rand(onum, fvnum)) * sqrt(6 / (onum + fvnum));
end