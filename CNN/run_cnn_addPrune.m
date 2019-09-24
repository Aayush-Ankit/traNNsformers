function cnn = run_cnn_addPrune(data_name, dataset_pathid, cnn, epochs, prune_slowdown, prunemode)

% Global file to keep traces (for debugging)
global fid;

%% Load the trained CNN if already trained
%load 'cnn_99.14.mat'

%% Load paths
% path if running on windows
% addpath(genpath('U:/AA/AproxSNN-ControlledSparsity/Matlab/'));
% path if running on linux
addpath(genpath('CNN_withPrune/'));

%% Load data
rand('state', 0);
load (dataset_pathid);
train_x = double(reshape(train_x',28,28,60000)) / 255;
train_y = double(train_y');
test_x = double(reshape(test_x',28,28,10000)) / 255;
test_y = double(test_y');
% Control number of testing images
% train_x = train_x(:,:,(1:10000));
% train_y = train_y(:,(1:10000));
test_x = test_x(:,:,(1:10000)); 
test_y = test_y(:,(1:10000));



%% Initialize net

% Type of layers: 
% 'c' for convolutional, 
% 's' for scalling/pooling
% 'f' for fully connected
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)  %convolution layer
    struct('type', 's', 'scale', 2)                         %sub sampling layer
    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 5)  %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
    struct('type', 'f', 'size', 1200)
    struct('type', 'f', 'size', 1200)
    struct('type', 'f', 'size', size(test_y, 1))
};

cnn = cnnsetup_addPrune(cnn, train_x, train_y);

% Set the activation function to be a ReLU
cnn.act_fun = @(inp)max(0, inp);
% Set the derivative to be the binary derivative of a ReLU
cnn.d_act_fun = @(forward_act)double(forward_act>0);


%% ReLU settings
% Set up learning constants
opts.alpha = 0.5;                       % learning rate
opts.momentum = 0.5;                    % momentum
opts.batchsize = 400;                    % batchsize
opts.numepochs =  epochs;               % epochs
opts.learn_bias = 0;                    % bias
opts.dropout = 0.0;
cnn.first_layer_dropout = 0;

% NEW
cnn.scaling_learningRate = 1;
cnn.weightPenaltyL2 = 0;
cnn.nonSparsityPenalty = 0;
cnn.sparsityTarget = 0.05;
cnn.inputZeroMaskedFraction = 0;
cnn.dropoutFraction = 0.5;
cnn.testing = 0;


%% TraNNsformer constants
% TRANNSFORMER
    cnn.clusterstartepoch = 0.2* opts.numepochs; % epoch when clustering map is created (need to start from somewhat pruned map)
    cnn.prunemode = prunemode;
    cnn.scaling_pruneRate = prune_slowdown * 0.001*[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]; % prune_slowdown is an external parameter
    cnn.utilth = 0.01*[70 70 70 70 70 70 70 70];
    cnn.crossbarSize = 64;
    cnn.tol = 0.05; % delta_unclustered synpases when pruning should be stopped
    cnn.cluster_base_quality_max = 0.7;
    cnn.cluster_base_quality_min = 0.3;
    cnn.cluster_prune_start = 0;
    cnn.cluster_prune_start = 0.8* opts.numepochs;
    cnn.scale_clusterpruneRate = [0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01];

assert(size(cnn.scaling_pruneRate, 2) == cnn.n, 'Size of scaling_pruneRate variable does not match with number of layers')
assert(size(cnn.utilth, 2) == cnn.n, 'Size of utilization variable does not match with number of layers')
assert(size(cnn.scale_clusterpruneRate, 2) == cnn.n, 'Size of scale_clusterpruneRate variable does not match with number of layers')


%% Create record file

kernel1 = cnn.layers{2}.outputmaps;
kernel2 = cnn.layers{4}.outputmaps;
cnn.FCcounter = 0;
cnn.firstFClayerIndex = 0;
for i = 1:numel(cnn.layers)
    if (cnn.layers{i}.type == 'f')
        if (cnn.FCcounter == 0)
            cnn.firstFClayerIndex = i;
        end
        cnn.FCcounter = cnn.FCcounter + 1;
    end
end

fid = fopen([sprintf('CNNoutputs/%s/trace_prunemode%d_numlayers%d_with%dFClayers_numfilters%dand%d_learningRate%2.2f_xbarutilmin%0.2f',...
    data_name, prunemode, cnn.n-1, cnn.FCcounter, kernel1, kernel2, opts.alpha, cnn.cluster_base_quality_min) '.txt'],'w');
fprintf(fid, [dataset_pathid, '\n']);



%% Initial Hardware estimations
% TRANNSFORMER
[num_mpe, ~] = get_hardware_params_cnn(cnn, 0);
fprintf(fid, 'Number of mPEs needed before trannsformation: %d\n', num_mpe);


%% Train

% Train - takes about 199 seconds per epoch on my machine - (for 16,16 conv layers)
cnn = cnntrain_addPrune(cnn, train_x, train_y, opts, test_x, test_y);

%% Test

[er, train_bad] = cnntest_addPrune(cnn, train_x, train_y);
fprintf(fid,'TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = cnntest_addPrune(cnn, test_x, test_y);
fprintf(fid,'TEST Accuracy: %2.2f%%.\n', (1-er)*100);


%% Plot the cluster quality histograms after SCIC & pruned fractions

if (cnn.prunemode == 2)
if_hist = 1;
    % find the updated clustering statistics & plot them (histograms)
    fig = figure(1);
    p = 1;
    for i = (cnn.firstFClayerIndex) : (cnn.n)
        prunestats = 100* sum(sum(cnn.map{i}))/(size(cnn.map{i},1) * size(cnn.map{i},2));
        fprintf(fid, 'Pruned percentage of Layer %d: %2.2f%%.\n', i, 100-prunestats);
        subplot(1,cnn.FCcounter,p)
        p = p + 1;
        % final connectivity matrix - logic or of cmap and pmap
        conn_matrix = logical(cnn.cmap{i}) | logical(cnn.pmap{i});
        cnn_analyse_cluster(cnn.clusters{i}, conn_matrix, if_hist);
    end
    saveas(fig, sprintf('CNNoutputs/%s/hist_prunemode%d_numlayers%d_with%dFClayers_xbarutilmin%0.2f.png',...
        data_name, prunemode, (cnn.n-1), cnn.FCcounter, cnn.cluster_base_quality_min))
end

%% Saving Output Data

save ([sprintf('CNNoutputs/%s/trace_prunemode%d_numlayers%d_with%dFClayers_numfilters%dand%d_learningRate%2.2f_xbarutilmin%0.2f',... 
    data_name, prunemode, (cnn.n-1), cnn.FCcounter, kernel1, kernel2, opts.alpha, cnn.cluster_base_quality_min) '.mat'],'cnn','opts');


    %% Extract the hardware results
    [num_mpe, ~, num_mpe_unclustered] = get_hardware_params_cnn(cnn, prunemode);
    fprintf(fid, 'Training effort in terms of number of epochs: %d\n', epochs);
    fprintf(fid, 'Total Number of mPEs needed after transformation: %d\n', num_mpe);
    fprintf(fid, 'Number of mPEs needed for unclustered synapses: %d\n', num_mpe_unclustered);
    fclose(fid);

end