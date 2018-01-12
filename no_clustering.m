%% Get the results from no Clustering (prunemode = 3)
clear; close all; clc
warning('off','all')

% parameters used to generate histograms - saved for reproducibility
mnist_cluster_base_quality_min = 0.0001;
svhn_cluster_base_quality_min = 0.0001;
cifar10_cluster_base_quality_min = 0.0001;

%% Cluster no - study and save the cluster results
% Load nn
data_name = 'cifar10';
% path where trained nns are stored
path_id = '/home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Results/Algorithm/%s/nn_prunemode1.mat';
nnpath_id = sprintf (path_id, data_name);
load (nnpath_id);

fid = fopen(sprintf('output/no_clustering/%s/trace.txt', data_name), 'w');
% cluster no
nn.cluster_base_quality_min = mnist_cluster_base_quality_min; %decides which quality clusters to find

qual_vec = {};
fig = figure(1);
for i = 1:nn.n-1
    fprintf('Current layer being processed: %d\n', i);
    % adjust the map size wrt crossbar size
    xbar_size = nn.crossbarSize;
    pmap_t = nn.pmap{i};
    x_size = ceil(size(pmap_t,1)/xbar_size);
    y_size = ceil(size(pmap_t,2)/xbar_size);
    map = zeros(x_size*xbar_size, y_size*xbar_size);
    map((1:size(pmap_t,1)), (1:size(pmap_t,2))) = pmap_t;
    
    qual_vec_t = zeros(x_size*y_size,1);
    % traverse the map
    idx = 1;
    for j = 1:nn.crossbarSize:x_size
        for k = 1:nn.crossbarSize:y_size
            map_t = map(j:j+nn.crossbarSize-1, k:k+nn.crossbarSize-1);
            qual_vec_t(idx) = sum(sum(map_t))/(nn.crossbarSize^2);
            idx = idx + 1;
        end
    end
    
    % save the quality vector and olt the current layer's quality histogram
    qual_vec{i} = qual_vec_t;
    subplot(1,nn.n-1,i),
    histogram(qual_vec_t);
end

% save the qual_vec metadata for combine histogram
cifar10_pm1 = qual_vec;
save('qual_vec_data.mat', 'cifar10_pm1', '-append')
% save the histogram and cmap
saveas(fig, sprintf('output/no_clustering/%s/hist.png', data_name))
