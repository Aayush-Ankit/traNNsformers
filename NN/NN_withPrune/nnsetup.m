function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    
    % new addition - pruneth increment rate & crossbar_size for clustering
    % constraint
    nn.clusters = {}; % saves the formed clusters if any
    nn.prunemode = 1;
    nn.scaling_pruneRate = 0.05;
    nn.utilth = 0.8;
    nn.crossbarSize = 64;
    
    for i = 2 : nn.n   
        % weights and weight momentum
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));
        
        % new addition - prune threshold & prune map
        nn.pruneth{i-1} = 0;
        nn.pmap{i-1} = ones(nn.size(i), nn.size(i - 1));
        
        % new addition - cluster prune factor
        nn.cluster_prune_factor{i-1} = 0.005;
        
        % new addition - cluster threshold & cluster map
        nn.cmap{i-1} = ones(nn.size(i), nn.size(i - 1));
        
        % new addition - overall map
        nn.map{i-1} = ones(nn.size(i), nn.size(i - 1));
    end
    
    % initalize layer-wise prune_curr & prune_prev
    for i = 1 : (nn.n-1) 
        nn.unclustered_prev{i} = 0;
        nn.unclustered_curr{i} = 0;
    end
    
    % track cluster_count - only for debug
    for i = 1 : (nn.n-1) 
        nn.cluster_count{i} = 0;
    end
    
    
end
