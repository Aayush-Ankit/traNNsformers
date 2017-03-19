function nn = run_fcn (data_name, dataset_pathid, net, epochs, prune_slowdown)

    % A global file to store the rraces while exectution/training
    global fid;

    %% Load the trained NN if already trained
    %load 'mlp_mnist.mat'
    
    %% Load paths
    % path if running on windows
    % addpath(genpath('U:/AA/AproxSNN-ControlledSparsity/Matlab/'));
    % path if running on linux
    addpath(genpath('/home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Matlab/'));
    
    %% Load data
    rand('state', 0);
    % path if running on windows
    % load ('U:/AA/ReSpArch-SpintronicSNNProcessor_DAC_2017/Matlab/spiking_relu_conversion-master/dlt_cnn_map_dropout_nobiasnn/data/mnist_uint8.mat');
    % path if running on linux
    load (dataset_pathid);
    train_x = double(train_x) / 255;
    train_y = double(train_y);
    test_x  = double(test_x)  / 255;
    test_y  = double(test_y);

    test_x = test_x((1:1000),:);
    test_y = test_y((1:1000),:);warning('off','all')


    %% Initialize net
    nn = nnsetup(net);
    % Rescale weights for ReLU
    for i = 2 : nn.n   
        % Weights - choose between [-0.1 0.1]
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
        nn.vW{i - 1} = zeros(size(nn.W{i-1}));
    end
    
    %% ReLU Train
    % Set up learning constants
    nn.activation_function = 'relu';
    nn.output = 'relu';
    nn.learningRate = 0.5;
    nn.momentum = 0.5;
    nn.dropoutFraction = 0.5;
    nn.learn_bias = 0;
    opts.numepochs =  epochs;
    opts.batchsize = 400;
    nn.clusterstartepoch = 0.2* opts.numepochs; % epoch when clustering map is created (need to start from somewhat pruned map)
    nn.prunemode = 2;
    nn.scaling_pruneRate = prune_slowdown * 0.001*[0.5 0.5 3]; % prune_slowdown is an external parameter
    nn.utilth = 0.01*[70 70 70];
    nn.crossbarSize = 64;
    nn.tol = 0.05; % delta_unclustered synpases when pruning should be stopped
    nn.cluster_base_quality_max = 0.7;
    nn.cluster_base_quality_min = 0.3;
    nn.cluster_prune_start = 0;
    nn.cluster_prune_start = 0.9* opts.numepochs;
    nn.cluster_prune_factor = 0.05;
    nn.cluster_prune_acc_loss = 0.6; % in percentage

    %% Training with Iterative Clustering + Pruning
    fid = fopen(sprintf('output/%s/trace.txt', data_name), 'w');
    fprintf(fid, dataset_pathid);
    % Train - takes about 15 seconds per epoch on my machine
    nn = nntrain(nn, train_x, train_y, opts);
    % Test - should be 98.62% after 15 epochs
    [er, train_bad] = nntest(nn, train_x, train_y);
    fprintf(fid, 'TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
    [er, bad] = nntest(nn, test_x, test_y);
    fprintf(fid, 'Test Accuracy: %2.2f%%.\n', (1-er)*100);

    % Plot the cluster quality histograms after SCIC & pruned fractions
    if_hist = 1;
    if (nn.prunemode == 2)
        % find the updated clustering statistics & plot them (histograms)
        fig = figure(1);
        for i = 1:nn.n-1
            prunestats = 100* sum(sum(nn.map{i}))/(size(nn.map{i},1) * size(nn.map{i},2));
            fprintf(fid, 'Pruned percentage of Layer %d: %2.2f%%.\n', i, 100-prunestats);
            subplot(1,nn.n-1,i),
            % final connectivity matrix - logic or of cmap and pmap
            conn_matrix = logical(nn.cmap{i}) | logical(nn.pmap{i});
            analyse_cluster(nn.clusters{i}, conn_matrix, if_hist);
        end
        saveas(fig, sprintf('output/%s/hist.png', data_name))
    end

    fclose(fid);
    save (sprintf('output/%s/nn.mat', data_name), 'nn', 'opts');
    
end
