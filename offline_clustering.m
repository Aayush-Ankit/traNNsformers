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
data_name = 'cifar10';
% path where trained nns are stored
path_id = '/home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Results/Algorithm/%s/nn_prunemode1.mat';
nnpath_id = sprintf (path_id, data_name);
load (nnpath_id);

fid = fopen(sprintf('output/offline_clustering/%s/trace.txt', data_name), 'w');
% cluster offline
nn.cluster_base_quality_min = cifar10_cluster_base_quality_min; %decides which quality clusters to find

for i = 1:nn.n-1
    fprintf(fid, 'Layer: %d Clustering of unclustered synapses being tried...\n', i); % for debug only
    % get the unclustered conn_matrix (conn_mat_uc)
    nn.clusters{i}.size = 0;
    conn_mat_uc = nn.pmap{i};

    % run clustering now on unclustered synapses
    clusters = {};
    clusters.size = 0;
    base_quality = nn.cluster_base_quality_max + 0.1;
    while (clusters.size == 0)
        base_quality = base_quality - 0.1;
        if (base_quality <= nn.cluster_base_quality_min)
            break; 
        end
        fprintf(fid, 'Base_quality tried for clustering: %f\n', base_quality);
        clusters = size_constrained_cluster(conn_mat_uc, nn.crossbarSize, base_quality);
        
        % augment the new found clusters for ith layer to older ones
        if (clusters.size ~= 0)
            disp('New clusters found')
            nn = add_clusters (nn, i, clusters);
            % get the conn_matrix, clusters and then proceed with cdp
            conn_matrix = logical(nn.cmap{i}) | logical(nn.pmap{i}); % conn_matrix is union of pmap and cmap
            clusters = nn.clusters{i};
            nn.cmap{i} = cluster_createmap(conn_matrix, nn.utilth(i), clusters); % cmap gets updated with new clusters
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
for i = 1:nn.n-1
    prunestats = 100* sum(sum(nn.pmap{i}))/(size(nn.pmap{i},1) * size(nn.pmap{i},2));
    fprintf(fid, 'Pruned percentage of Layer %d: %2.2f%%.\n', i, 100-prunestats);
    subplot(1,nn.n-1,i),
    % final connectivity matrix - logic or of cmap and pmap
    conn_matrix = logical(nn.cmap{i}) | logical(nn.pmap{i});
    qual_vec{i} = analyse_cluster_mod(nn.clusters{i}, conn_matrix, if_hist);
end
cifar10_pm3 = qual_vec;

% save the qual_vec metadata for combine histogram
save('qual_vec_data.mat', 'cifar10_pm3', '-append')

% save the histogram and cmap
saveas(fig, sprintf('output/offline_clustering/%s/hist.png', data_name))
save (sprintf('output/offline_clustering/%s/nn.mat', data_name), 'nn', 'opts');

% save tha harwdare parameters
prunemode = 2; % (similar in style to onlice clustering)
[num_mpe, ~, num_mpe_unclustered] = get_hardware_params(nn, prunemode);
fprintf(fid, 'Total Number of mPEs needed after transformation: %d\n', num_mpe);
fprintf(fid, 'Number of mPEs needed for unclustered synapses: %d\n', num_mpe_unclustered);
