function cnn = cnn_prunewt(cnn, current_epoch)
%PRUNEWT performs pruning based on prune mode and prune maps (accuracy_map
%and cluster_map)
%nn = prunewt(nn) returns a neural network strcuture with pruned weights
%based on the current prune threshold, crossbar utilization threshold & updates the two maps accordingly
    
    global fid;

    cstart = cnn.clusterstartepoch;
    tol = cnn.tol; %for debug - 0.01 (shud be)
    n = cnn.n;
    
    for i = cnn.firstFClayerIndex : n
        % accuracy + clustering dependent pruning
        if (cnn.prunemode == 2)
            % no cluster map
            if (current_epoch < cstart)
                cnn = cnn_adp(cnn, i);
            % create the cluster map (for the first time) and use updated cmap and updated pmap for pruning
            elseif (current_epoch == cstart)
                conn_matrix = cnn.pmap{i};
                base_quality = cnn.cluster_base_quality_max;
                fprintf(fid, 'Base_quality for clustering: %f\n', base_quality);
                cnn.clusters{i} = cnn_size_constrained_cluster(conn_matrix, cnn.crossbarSize, base_quality);
                clusters = cnn.clusters{i};
                cnn.cmap{i} = cnn_cluster_createmap(conn_matrix, cnn.utilth(i), clusters);
                cnn.unclustered_prev{i} = cnn.unclustered_curr{i};
                cnn = cnn_cdp(cnn, i);
            % old cmap and updated pmap is used for pruning
            % re-cluster the layer if the change in pruned connections is
            % less than tolerance
            else
                fprintf(fid, 'Layer: %d \t unclustered current: %f \t unclustered previous: %f\n', i, cnn.unclustered_curr{i}, cnn.unclustered_prev{i}); 
                if (abs(cnn.unclustered_curr{i} - cnn.unclustered_prev{i}) >= tol)
                    fprintf(fid, 'Pruning continues...\n'); % for debug only
                    cnn.unclustered_prev{i} = cnn.unclustered_curr{i};
                    cnn = cnn_cdp(cnn, i);
                % try to cluster the currently unclustered synapses - Note
                % pruning of unclustered should be given more importance
                % than clustering
                else
                    fprintf(fid, 'Clustering of unclustered synapses being tried...\n'); % for debug only
                    % get the unclustered conn_matrix (conn_mat_uc)
                    conn_mat_uc = cnn.pmap{i};
                    conn_mat_uc(cnn.cmap{i} == 1) = 0;
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
                    end
                    
                    % augment the new found clusters for ith layer to older ones
                    if (clusters.size ~= 0)
                        cnn = cnn_add_clusters (cnn, i, clusters);
                        % get the conn_matrix, clusters and then proceed with cdp
                        conn_matrix = logical(cnn.cmap{i}) | logical(cnn.pmap{i}); % conn_matrix is union of pmap and cmap
                        clusters = cnn.clusters{i};
                        cnn.cmap{i} = cnn_cluster_createmap(conn_matrix, cnn.utilth(i), clusters); % cmap gets updated with new clusters
                    end
                    
                    cnn.unclustered_prev{i} = cnn.unclustered_curr{i};
                    fprintf(fid, 'Pruning continues...\n'); % for debug only
                    cnn = cnn_cdp(cnn, i);
                end
            end
               
        % accuracy dependent pruning only
        elseif (cnn.prunemode == 1)
            cnn = cnn_adp(cnn, i);
        end
    end
    
end