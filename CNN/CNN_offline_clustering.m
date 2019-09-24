%% Get the results from Offline Clustering (prunemode = 3)
clear; close all; clc
warning('off','all')

global fid;

% parameters used to generate histograms - saved for reproducibility
mnist_cluster_base_quality_min = 0.0001;
svhn_cluster_base_quality_min = 0.2;
cifar10_cluster_base_quality_min = 0.0001;

%% Cluster Offline
% Load nn
data_name = 'mnist';
% path where trained nns are stored
addpath(genpath('CNN_withPrune/'));
path_id = 'CNNoutputs/%s/trace_prunemode1_numlayers7_with3FClayers_numfilters12and64_learningRate0.70_xbarutilmin0.30.mat';
nnpath_id = sprintf (path_id, data_name);
load (nnpath_id);


fid = fopen(sprintf('CNNoutputs/%s/resultset2_offline_clustering/trace.txt', data_name), 'w');
% cluster offline
cnn.cluster_base_quality_min = mnist_cluster_base_quality_min; %decides which quality clusters to find

fprintf(fid, 'Start of offline clustering!\n');
for i = cnn.firstFClayerIndex : cnn.n
    fprintf(fid, 'Layer: %d Clustering of unclustered synapses being tried...\n', i); % for debug only
    % get the unclustered conn_matrix (conn_mat_uc)
    cnn.clusters{i}.size = 0;
    conn_mat_uc = cnn.pmap{i};

    % run clustering now on unclustered synapses
    clusters = {};
    clusters.size = 0;
    base_quality = cnn.cluster_base_quality_max + 0.1;
    while (clusters.size == 0)
        base_quality = base_quality - 0.1;
        if (base_quality <= cnn.cluster_base_quality_min)
            break; 
        end
        fprintf(fid, 'Base_quality tried for clustering: %f\n', base_quality);
        clusters = cnn_size_constrained_cluster(conn_mat_uc, cnn.crossbarSize, base_quality);
        
        % augment the new found clusters for ith layer to older ones
        if (clusters.size ~= 0)
            disp('New clusters found')
            cnn = cnn_add_clusters (cnn, i, clusters);
            % get the conn_matrix, clusters and then proceed with cdp
            conn_matrix = logical(cnn.cmap{i}) | logical(cnn.pmap{i}); % conn_matrix is union of pmap and cmap
            clusters = cnn.clusters{i};
            cnn.cmap{i} = cnn_cluster_createmap(conn_matrix, cnn.utilth(i), clusters); % cmap gets updated with new clusters
        end
        
        %reset cluster size
        clusters.size = 0;
    end
end

%% Plot the cluster quality histograms after SCIC & pruned fractions
qual_vec = {};
if_hist = 1;
% find the updated clustering statistics & plot them (histograms)
fig = figure(1);
p = 1;
fprintf(fid, '*******RESULTS OF offline clustering:*******\n');

for i = cnn.firstFClayerIndex : cnn.n
    prunestats = 100* sum(sum(cnn.pmap{i}))/(size(cnn.pmap{i},1) * size(cnn.pmap{i},2));
    fprintf(fid, 'Pruned percentage of Layer %d: %2.2f%%.\n', i, 100-prunestats);
    subplot(1,cnn.FCcounter,p)
    p = p + 1;
    % final connectivity matrix - logic or of cmap and pmap
    conn_matrix = logical(cnn.cmap{i}) | logical(cnn.pmap{i});
    qual_vec{i} = cnn_analyse_cluster_mod(cnn.clusters{i}, conn_matrix, if_hist);
end
mnist_pm3 = qual_vec;

% save the qual_vec metadata for combine histogram
save('CNNoutputs/mnist/qual_vec_data.mat', 'mnist_pm3')

% save the histogram and cmap
saveas(fig, sprintf('CNNoutputs/%s/resultset2_offline_clustering/hist.png', data_name))
save (sprintf('CNNoutputs/%s/resultset2_offline_clustering/cnn.mat', data_name), 'cnn', 'opts');

% save tha harwdare parameters
prunemode = 2; % (similar in style to onlice clustering)
[num_mpe, ~, num_mpe_unclustered] = get_hardware_params_cnn(cnn, prunemode);
fprintf(fid, 'Total Number of mPEs needed after transformation: %d\n', num_mpe);
fprintf(fid, 'Number of mPEs needed for unclustered synapses: %d\n', num_mpe_unclustered);


%% Extract qual_vec results from the cluster analysis for online clustering to plot the combined histograms
% Load nn
data_name = 'mnist';
% path where trained nns are stored
path_id = 'CNNoutputs/%s/trace_prunemode2_noPruneSlowdown_earlyClusterPrune_numlayers7_with3FClayers_numfilters12and64_learningRate0.70_xbarutilmin0.30.mat';
nnpath_id = sprintf (path_id, data_name);
load (nnpath_id);

fprintf(fid, '*******RESULTS OF TraNNsformer framework:*******\n');
qual_vec = {};
if_hist = 1;
% find the updated clustering statistics & plot them (histograms)
fig = figure(1);
p = 1;
for i = cnn.firstFClayerIndex : cnn.n
    subplot(1,cnn.FCcounter,p)
    p = p + 1;
    % final connectivity matrix - logic or of cmap and pmap
    conn_matrix = logical(cnn.cmap{i}) | logical(cnn.pmap{i});
    qual_vec{i} = cnn_analyse_cluster_mod(cnn.clusters{i}, conn_matrix, if_hist);
end
mnist_pm2 = qual_vec;

% save the qual_vec metadata for combine histogram
save('CNNoutputs/mnist/qual_vec_data.mat', 'mnist_pm2', '-append')
