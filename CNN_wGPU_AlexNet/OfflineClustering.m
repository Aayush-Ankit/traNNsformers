addpath(genpath('pruneANDclusterFiles/'));
prunemode = 1;
global fid
fid = fopen([sprintf('CNNoutputs/AlexNet/trace_mPEs_prunemode%d_offlineClustering', prunemode) '.txt'],'w');

load('/data/tibrayev/traNNsformer_for_GPU_v2/checkpoints/prunemode1/epoch90/pruneANDoptsFiles.mat')
load('/data/tibrayev/traNNsformer_for_GPU_v2/checkpoints/prunemode1/epoch90/prunedNET.mat')
opts.mnist_cluster_base_quality_min = 0.0001;

for i = 1:opts.numberFCLayers
    fprintf(fid, 'Layer: %d Clustering of unclustered synapses being tried...\n', i); % for debug only
    % get the unclustered conn_matrix (conn_mat_uc)
    prune.clusters{i}.size = 0;
    conn_mat_uc = prune.pmap{i};

    % run clustering now on unclustered synapses
    clusters = {};
    clusters.size = 0;
    base_quality = opts.cluster_base_quality_max + 0.1;
    while (clusters.size == 0)
        base_quality = base_quality - 0.1;
        if (base_quality <= opts.mnist_cluster_base_quality_min)
            break; 
        end
        fprintf(fid, 'Base_quality tried for clustering: %f\n', base_quality);
        clusters = cnn_size_constrained_cluster(conn_mat_uc, opts.crossbarSize, base_quality);
        
        % augment the new found clusters for ith layer to older ones
        if (clusters.size ~= 0)
            disp('New clusters found')
            prune = cnn_add_clusters (prune, i, clusters);
            % get the conn_matrix, clusters and then proceed with cdp
            conn_matrix = logical(prune.cmap{i}) | logical(prune.pmap{i}); % conn_matrix is union of pmap and cmap
            clusters = prune.clusters{i};
            prune.cmap{i} = cnn_cluster_createmap(conn_matrix, opts.utilth(i), clusters); % cmap gets updated with new clusters
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
for i = 1:opts.numberFCLayers
    prunestats = 100* sum(sum(prune.pmap{i}))/(size(prune.pmap{i},1) * size(prune.pmap{i},2));
    fprintf(fid, 'Pruned percentage of Layer %d: %2.2f%%.\n', i, 100-prunestats);
    subplot(1,opts.numberFCLayers,p)
    p = p + 1;
    % final connectivity matrix - logic or of cmap and pmap
    conn_matrix = logical(prune.cmap{i}) | logical(prune.pmap{i});
    qual_vec{i} = cnn_analyse_cluster_mod(prune.clusters{i}, conn_matrix, if_hist);
    axis([0 1 0 inf])
end
mnist_pm3 = qual_vec;

% save the qual_vec metadata for combine histogram
save('qual_vec_data.mat', 'mnist_pm3')

% save the histogram and cmap
saveas(fig, sprintf('CNNoutputs/AlexNet/hist_prunemode%d_offlineClustering_xbarutilmin%0.2f.png',...
        opts.prunemode, opts.mnist_cluster_base_quality_min))
save(sprintf('CNNoutputs/AlexNet/prunemode%d_offlineClusteringData.mat', opts.prunemode), 'cnn_alexnet_prune', 'opts', 'prune', 'optsTrain');

% save tha harwdare parameters
prunemode = 2; % (similar in style to online clustering)
[num_mpe, ~, num_mpe_unclustered] = get_hardware_params_cnn(cnn_alexnet_prune, opts, prune, prunemode);
fprintf(fid, 'Total Number of mPEs needed after transformation: %d\n', num_mpe);
fprintf(fid, 'Number of mPEs needed for unclustered synapses: %d\n', num_mpe_unclustered);