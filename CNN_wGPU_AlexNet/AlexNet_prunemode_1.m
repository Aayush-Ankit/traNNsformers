%% TraNNsformer Code for AlexNet CNN to run on GPU (given one or any is available)
% Things TO DO:
% 1. Create and Train AlexNet on its own
% [DONE: for two epochs. Each epoch took approximately 7hrs on 3 gpu's.
% Results are stored in /checkpoints/firstTrial]
%
% 2. Train AlexNet with each Epoch being fetched separately -> the main
% task will be to preserve weights (and overall structure) between
% transitions from one epoch to another epoch WITHOUT accuracy degradation!
% [NOT DONE, BUT Task 1 provided required information]
%
% 3. If task 2 is successful, add Pruning between consecutive epochs! ->
% the main task here will be to incorporate moving weights to GPU (gpuArray), perform
% Prune computation on GPU, and, if necessary, to move weights back to CPU!
% 
% NOTE: After task 1 and task 2 check if the weights are already on GPU in
% the format like gpuArray (????)
% 
% 4. Train AlexNet with Prunemode 1 with required additions to preserve
% accuracy results
%
%% Pruning settings

prunemode = 1;
addpath(genpath('pruneANDclusterFiles/'));

epochs = 300;
epochs_reference = 150;
prune_slowdown = epochs / epochs_reference;

opts.prunemode = prunemode;
opts.numepochs = epochs;
opts.prune_slowdown = prune_slowdown;
opts.minibatchSize = 60;
opts.numberOfWorkers = gpuDeviceCount;

CONTINUATION = 1;

%% Creating trace file

global fid
%global MiniBatchLosses
global averageEpochLoss

if CONTINUATION == 0
    fid = fopen([sprintf('CNNoutputs/AlexNet/trace_prunemode%d', prunemode) '.txt'],'w');
elseif CONTINUATION == 1
    fid = fopen([sprintf('CNNoutputs/AlexNet/trace_prunemode%d', prunemode) '.txt'],'a'); % For appending to existing file
end
diary(sprintf('CNNoutputs/AlexNet/trace_prunemode%d_NNtoolbox_%s_%d.txt', prunemode, datestr(now,'yyyy_mm_dd_HH_MM_SS'),randi([1,10000],1)))

%% Load ImageNet dataset

tic
fprintf(fid, ['Data loading started ...', '\n']);
fprintf(['Data loading started ...', '\n']);
imdsTrain = imageDatastore('/data/tibrayev/imagenet2012_zeromeanNormalized/train', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');
imdsTest  = imageDatastore('/data/tibrayev/imagenet2012_zeromeanNormalized/val', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');
% Resize
% imdsTrain.ReadFcn = @(filename)readAndPreprocessImage(filename);
% imdsTest.ReadFcn = @(filename)readAndPreprocessImage(filename);
t = toc;
fprintf(fid, ['Data loading completed. Time taken: ', num2str(t), ' seconds.', '\n']);
fprintf(['Data loading completed. Time taken: ', num2str(t), ' seconds.', '\n']);

%% Create AlexNet CNNetwork

tic
fprintf(fid, ['Network structuring started ...', '\n']);
fprintf(['Network structuring started ...', '\n']);
% AlexNet Structure
cnn_alexnet_structure = [
    imageInputLayer([227 227 3], 'Name', 'data', 'Normalization', 'none')
    convolution2dLayer(11, 96, 'NumChannels', 3, 'Stride', 4, 'BiasLearnRateFactor', 2, 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    crossChannelNormalizationLayer(5,'K',1, 'Name', 'norm1')
    maxPooling2dLayer([3 3], 'Stride', 2, 'Name', 'pool1')
    convolution2dLayer(5, 256, 'Stride', 1, 'Padding', 2, 'BiasLearnRateFactor', 2, 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    crossChannelNormalizationLayer(5,'K',1, 'Name', 'norm2')
    maxPooling2dLayer([3 3], 'Stride', 2, 'Name', 'pool2')
    convolution2dLayer(3, 384, 'NumChannels', 256, 'Stride', 1, 'Padding', 1, 'BiasLearnRateFactor', 2, 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1, 'BiasLearnRateFactor', 2, 'Name', 'conv4')
    reluLayer('Name', 'relu4')
    convolution2dLayer(3, 256, 'Stride', 1, 'Padding', 1, 'BiasLearnRateFactor', 2, 'Name', 'conv5')
    reluLayer('Name', 'relu5')
    maxPooling2dLayer([3 3], 'Stride', 2, 'Name', 'pool5')
    fullyConnectedLayer(4096, 'BiasLearnRate', 2, 'Name','fc6')
    reluLayer('Name', 'relu6')
    dropoutLayer('Name', 'drop6')
    fullyConnectedLayer(4096, 'BiasLearnRate', 2, 'Name','fc7')
    reluLayer('Name', 'relu7')
    dropoutLayer('Name', 'drop7')
    fullyConnectedLayer(1000, 'BiasLearnRate', 2, 'Name','fc8')
    softmaxLayer('Name', 'prob')
    classificationLayer('Name', 'output')];
t = toc;
fprintf(fid, ['Network structuring completed. Time taken: ', num2str(t), ' seconds.', '\n']);
fprintf(['Network structuring completed. Time taken: ', num2str(t), ' seconds.', '\n']);

%% Network dependent pruning options
opts.fcIndices = [];
for i = 1:numel(cnn_alexnet_structure)
    if strcmp(cnn_alexnet_structure(i).Name(1:2), 'fc')
        opts.fcIndices(end+1) = i;        
    end
end
opts.numberFCLayers = numel(opts.fcIndices);
assert(opts.numberFCLayers ~= 0, 'No FC layers found!')

%% Train

fprintf(fid, ['Training started ...', '\n']);
fprintf(['Training started ...', '\n']);

if CONTINUATION == 0
% Comment/Uncomment these if simulation is "from scratch"/"continuation"
    opts.tr_err_prev = 1;
    opts.pruneANDclusterStart = 0;
    loss.train.e = 10;
% AlexNet default training parameters
    opts.currentLearnRate = 0.01;
    opts.learnRateDropActive = 1;
    opts.lossTolerance = 0.01;
    opts.learnRateDropFactor = 0.1;
    opts.l2Regularization = 0.0005;
    opts.Momentum = 0.9;
    continuationEpoch = 1;
    
elseif CONTINUATION == 1
% IN CASE OF CONTINUATION:
    continuationEpoch = 80;
    load(sprintf('./checkpoints/prunemode%d/epoch%d/pruneANDoptsFiles.mat', prunemode, continuationEpoch));
    load(sprintf('./checkpoints/prunemode%d/epoch%d/prunedNET.mat', prunemode, continuationEpoch));
    %cnn_alexnet_prune = net.Layers;
end
    
%% Training Epochs
for e = continuationEpoch+1:opts.numepochs
    
    % Defining training options
    str_checkpoint = sprintf('./checkpoints/prunemode%d/epoch%d/', prunemode, e);
    mkdir(str_checkpoint)
    optsTrain = trainingOptions('sgdm', ...
        'MiniBatchSize', opts.minibatchSize*opts.numberOfWorkers, ...
        'MaxEpochs', 1, ...
        'InitialLearnRate', opts.currentLearnRate, ...
        'L2Regularization', opts.l2Regularization, ...
        'Momentum', opts.Momentum, ...
        'CheckpointPath', str_checkpoint, ...
        'OutputFcn', @(info)getMiniBatchLoss(info), ...
        'ExecutionEnvironment', 'multi-gpu');
    
    TEpochSTART = tic;
    fprintf(fid, ['****************START of Epoch: ', num2str(e) '/' num2str(opts.numepochs) '****************\n']);
    fprintf(['****************START of Epoch: ', num2str(e) '/' num2str(opts.numepochs) '****************\n']);
    if e == 1
        cnn_alexnet = trainNetwork(imdsTrain, cnn_alexnet_structure, optsTrain);
        cnn_alexnet_prune = cnn_alexnet.Layers;
        
        % Initialize pruning parameters
        for f = 1:opts.numberFCLayers
            % TRANNSFORMER
            % new addition - prune threshold & prune map
            prune.pruneth{f} = 0;
            prune.pmap{f} = ones(size(cnn_alexnet_prune(opts.fcIndices(f)).Weights));
            % new addition - cluster prune factor
            prune.cluster_prune_factor{f} = 0.005;
            % new addition - cluster threshold & cluster map
            prune.cmap{f} = ones(size(cnn_alexnet_prune(opts.fcIndices(f)).Weights));
            % new addition - overall map 
            prune.map{f} = ones(size(cnn_alexnet_prune(opts.fcIndices(f)).Weights));

            % initialize layer-wise prune-curr & prune_prev
            prune.unclustered_prev{f} = 0;
            prune.unclustered_curr{f} = 0;
            % track cluster_count for debug
            prune.cluster_count{f} = 0;
        end
        
    else
        cnn_alexnet = trainNetwork(imdsTrain, cnn_alexnet_prune, optsTrain);
        cnn_alexnet_prune = cnn_alexnet.Layers;
    end
    
    loss.train.e(end+1) = gather(averageEpochLoss);
    TEpochEND = toc(TEpochSTART);
    str_perf = sprintf('Average mini-batch train loss = %f', loss.train.e(end));
    disp_msg1 = [str_perf];
    fprintf(fid, '%s\n', disp_msg1);
    
    % Check if learn rate should be decreased (Condition is that the change
    % of loss is lower than opts.lossTolerance (default = 0.01))
    % NOTE that if learn rate drop is deactivated if pruning starts (This
    % is due to the fact that pruning will result in oscillations of loss
    % values)
    if (opts.learnRateDropActive == 1)
        if ((loss.train.e(end-1)-loss.train.e(end)) < opts.lossTolerance)
           opts.currentLearnRate = opts.currentLearnRate * opts.learnRateDropFactor;
           disp_msg1_2 = ['Learning Rate changed from ' num2str(opts.currentLearnRate/opts.learnRateDropFactor) ' to ' num2str(opts.currentLearnRate)];
           fprintf(fid, '%s\n', disp_msg1_2);
        else
           disp_msg1_1 = ['Current Learning Rate = ' num2str(opts.currentLearnRate)];
           fprintf(fid, '%s\n', disp_msg1_1);
        end
    end
    
    % Check if pruning can be started (Condition is that training loss
    % value drops below initial value of opts.tr_err_prev (default = 1))
    if (opts.tr_err_prev > loss.train.e(end) && opts.pruneANDclusterStart == 0)
        fprintf(fid, ['*****************Pruning is ACTIVATED*****************', '\n']);
        fprintf(fid, ['************Learn Rate Drop is DEACTIVATED************', '\n']);
        % TraNNsformer constants
        opts.learnRateDropActive = 0; % Stop learning rate drop if pruning started -> because pruning will result in oscillations of loss values
        opts.pruneANDclusterStart = e;
        opts.clusterstartepoch = ceil(0.3 * (opts.pruneANDclusterStart)) + opts.pruneANDclusterStart; % epoch when clustering map is created (need to start from somewhat pruned map)
        opts.scaling_pruneRate = prune_slowdown * 0.001*0.5 * ones(1, opts.numberFCLayers) ; % prune_slowdown is an external parameter
        opts.utilth = 0.01*70 * ones(1, opts.numberFCLayers);
        opts.crossbarSize = 64;
        opts.tol = 0.05; % delta_unclustered synpases when pruning should be stopped
        opts.cluster_base_quality_max = 0.7;
        opts.cluster_base_quality_min = 0.3;
        opts.cluster_prune_start = ceil(0.7 * (opts.pruneANDclusterStart)) + opts.pruneANDclusterStart; %(opts.numepochs - e);
        opts.scale_clusterpruneRate = 0.01 * ones(1, opts.numberFCLayers);

        fprintf(fid, ['*****************Prune parameters:*****************', '\n']);
        fprintf(fid, ['Prunining is activated at epoch ', num2str(opts.pruneANDclusterStart), '\n']);
        fprintf(fid, ['Prune slowdown: ', num2str(prune_slowdown), '\n']);
        if (prunemode == 2)
        fprintf(fid, ['Cluster start epoch: ', num2str(opts.clusterstartepoch), '\n']);
        fprintf(fid, ['Cluster prune start epoch: ', num2str(opts.cluster_prune_start), '\n']);
        end
        fprintf(fid, ['****************************************************', '\n']);
    end
    
    % Pruning and Clustering
    if (opts.pruneANDclusterStart ~= 0)
    TPruneSTART = tic;
% NEW addition - prune at the end of each epoch (uses both pmap and cmap) -
    % removes discrete synapses
    if ((opts.prunemode == 1) || (opts.prunemode == 2))
        [cnn_alexnet_prune, prune] = cnn_prunewt(cnn_alexnet_prune, e, opts, prune);
    end
    % cluster_prune every epoch (after cluster_prune_start epoch)
    if ((opts.prunemode == 2) && (e >= opts.cluster_prune_start))
        [cnn_alexnet_prune, prune] = cnn_cluster_prune(cnn_alexnet_prune, opts, prune);
    end
    
% NEW addition - increase the prune threshold of each layer after an epoch
    % If the tr_error decreases then
    % 1. increase the pruning threshold (discrete synapses)
    % 2. inclrease the cluster_prune threshold (clustered synapses)
    if ((opts.prunemode == 1) || (opts.prunemode == 2))
        if (opts.tr_err_prev > loss.train.e(end))
            fprintf(fid, 'increasing the pruning threshold err_prev = %0.4f\t err_crr = %0.4f\n', opts.tr_err_prev, loss.train.e(end));
            for p = 1:opts.numberFCLayers
                prune.pruneth{p} = prune.pruneth{p} + opts.scaling_pruneRate(p);
                if ((opts.prunemode == 2) && (e >= opts.cluster_prune_start)) % start cluster_pruning towards the later end of training
                    fprintf(fid, 'increasing the cluster pruning threshold\n');
                    prune.cluster_prune_factor{p} = prune.cluster_prune_factor{p} + opts.scale_clusterpruneRate(p);
                end
            end      
        end
        opts.tr_err_prev = loss.train.e(end); 
    end
    TPruneEND = toc(TPruneSTART);
    
    disp_msg2 = ['Pruning required time: ' num2str(TPruneEND) ' seconds.'];
    fprintf(fid, '%s\n', disp_msg2);
    fprintf(['Pruning required time: ' num2str(TPruneEND) ' seconds.' '\n']);
    end
    
    save([sprintf('./checkpoints/prunemode%d/epoch%d/pruneANDoptsFiles', prunemode, e) '.mat'], 'opts', 'prune', 'optsTrain', 'loss');
    save([sprintf('./checkpoints/prunemode%d/epoch%d/prunedNET', prunemode, e) '.mat'], 'cnn_alexnet_prune');
    

    disp_msg3 = ['Epoch: ' num2str(e) '/' num2str(opts.numepochs) '-----time: ' num2str(TEpochEND) ' seconds.'];
    fprintf(fid, '%s\n', disp_msg3);
    fprintf(['Epoch: ' num2str(e) '/' num2str(opts.numepochs) '-----time: ' num2str(TEpochEND) ' seconds\n']);
    
end

%%
fprintf(fid, ['Training completed.', '\n']);
fprintf(['Training completed.', '\n']);


%% Test

fprintf(fid, ['***************************************************', '\n']);
tic
fprintf(fid, ['Classification started ...', '\n']);
fprintf(['Classification started ...', '\n']);
predictedLabels = classify(cnn_alexnet, imdsTest, ...
    'MiniBatchSize', opts.minibatchSize, ...
    'ExecutionEnvironment', 'gpu');

accuracy = sum(predictedLabels == imdsTest.Labels)/numel(imdsTest.Labels)*100;
t = toc;
fprintf(fid, ['Classification completed. Testing time: ', num2str(t), ' seconds.', '\n']);
fprintf(['Classification completed. Testing time: ', num2str(t), ' seconds.', '\n']);
fprintf(fid, ['TEST Accuracy is ', num2str(accuracy), '%%.', '\n']);
fprintf(['TEST Accuracy is ', num2str(accuracy), '%%.', '\n']);
fprintf(fid, ['***************************************************', '\n']);


fprintf(fid, ['************Prune parameters:************', '\n']);
fprintf(fid, ['Prune slowdown: ', num2str(prune_slowdown), '\n']);
if (prunemode == 2)
    fprintf(fid, ['Cluster start epoch: ', num2str(opts.clusterstartepoch), '\n']);
    fprintf(fid, ['Cluster prune start epoch: ', num2str(opts.cluster_prune_start), '\n']);
end
fprintf(fid, ['***************************************************', '\n']);


%% Plot the cluster quality histograms after SCIC & pruned fractions

if (opts.prunemode == 2)
if_hist = 1;
    % find the updated clustering statistics & plot them (histograms)
    fig = figure(1);
    p = 1;
    for i = 1:opts.numberFCLayers
        prunestats = 100* sum(sum(prune.map{i}))/(size(prune.map{i},1) * size(prune.map{i},2));
        fprintf(fid, 'Pruned percentage of FC Layer %d: %2.2f%%.\n', i, 100-prunestats);
        subplot(1,opts.numberFCLayers,p)
        p = p + 1;
        % final connectivity matrix - logic or of cmap and pmap
        conn_matrix = logical(prune.cmap{i}) | logical(prune.pmap{i});
        cnn_analyse_cluster(prune.clusters{i}, conn_matrix, if_hist);
    end
    saveas(fig, sprintf('CNNoutputs/AlexNet/hist_prunemode%d_xbarutilmin%0.2f.png',...
        opts.prunemode, opts.cluster_base_quality_min))
end

%% Saving Output Data

save ([sprintf('CNNoutputs/AlexNet/trace_prunemode%d_xbarutilmin%0.2f',... 
    opts.prunemode, opts.cluster_base_quality_min) '.mat'], ...
    'cnn_alexnet', 'cnn_alexnet_prune', 'opts', 'prune', 'optsTrain');

%% Extract the hardware results
% Initial Hardware estimations
[num_mpe, ~] = get_hardware_params_cnn(cnn_alexnet_prune, opts, prune, 0);
fprintf(fid, 'Number of mPEs needed before trannsformation: %d\n', num_mpe);
% Final Hardware estimations
[num_mpe, ~, num_mpe_unclustered] = get_hardware_params_cnn(cnn_alexnet_prune, opts, prune, prunemode);
fprintf(fid, 'Training effort in terms of number of epochs: %d\n', epochs);
fprintf(fid, 'Total Number of mPEs needed after transformation: %d\n', num_mpe);
fprintf(fid, 'Number of mPEs needed for unclustered synapses: %d\n', num_mpe_unclustered);

diary off;
fclose(fid);