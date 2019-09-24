function [ ] = cnn_analyse_cluster(clusters, conn_matrix, if_hist)
%ANALYSE_CLUSTER analyzes the formed clusters and the original connectivity
%matrix for cluster utilizations
    
    global fid;

    % setup
    n = clusters.size;
    qual_vec = zeros(clusters.size,1); %stores the quality of all clusters
    num_synapses_cluster = 0;
    [h, k] = size(conn_matrix);
    base_q = sum(sum(conn_matrix)) / (h*k);
    cluster_map = zeros(h,k); % '1' tells that synapse is a part of a cluster
    
    % traverse all clusters
    idx = 0;
    for i = 1:n
        % ignore the clusters with Q = 0. (invalidated)
        if (clusters.C{i}.Q ~= 0)
            idx = idx + 1;
            qual_vec(idx) = clusters.C{i}.Q;
            cluster_map(clusters.C{i}.outputs, clusters.C{i}.inputs) = 1; % keeping only valid clusters - same as cmap
        end
        % cluster map keeps track of synapses that could be clustered (doesn't remove them if they got removed while cluster_pruning)
        % cluster_map(clusters.C{i}.outputs, clusters.C{i}.inputs) = 1;
    end
%     statistics - this is the fraction out of total synapses (not remaining unpruned synapses)
%     frac_unclustered_synapses = 1 - sum(sum(cluster_map .* conn_matrix))/ sum(sum(conn_matrix)); % conn_matrix the logical or of pmap and cmap
    % statistics - this is the fraction out of total synapses (not remaining unpruned synapses)
    unclus_map = (cluster_map == 0);
    unclus_map = unclus_map .* conn_matrix;
    [h,k] = size(unclus_map);
    %frac_unclustered_synapses = sum(sum(unclus_map))/ (h*k);
    frac_unclustered_synapses = sum(sum(unclus_map))/ (sum(sum(conn_matrix))); %<<== Correct way! Because fraction should be taken w.r.t. synapses left after pruning!
    
    fprintf(fid, 'Fraction of synapses unclustered = %0.4f\n', frac_unclustered_synapses);
    if (if_hist == 1)
        % only plot the hist of non-zero qual_ves (zero Q represents an invalidated cluster)
        histogram(qual_vec(1:idx))
        fprintf(fid, 'Number of clusters = %d\n', idx);
    end
    
end

