function [cnn, prune] = cnn_prunewt(cnn, current_epoch, opts, prune)
%PRUNEWT performs pruning based on prune mode and prune maps (accuracy_map
%and cluster_map)
%nn = prunewt(nn) returns a neural network strcuture with pruned weights
%based on the current prune threshold, crossbar utilization threshold & updates the two maps accordingly
    
    global fid;

    cstart = opts.clusterstartepoch;
    tol = opts.tol; %for debug - 0.01 (shud be)
    
    for i = 1 : opts.numberFCLayers
        % accuracy + clustering dependent pruning
        currentLayerWeights = cnn(opts.fcIndices(i)).Weights;
        if (opts.prunemode == 2)
            % no cluster map
            if (current_epoch < cstart)
                [currentLayerWeights, prune] = cnn_adp(currentLayerWeights, i, prune);
            % create the cluster map (for the first time) and use updated cmap and updated pmap for pruning
            elseif (current_epoch == cstart)
                conn_matrix = prune.pmap{i};
                base_quality = opts.cluster_base_quality_max;
                fprintf(fid, 'Base_quality for clustering: %f\n', base_quality);
                prune.clusters{i} = cnn_size_constrained_cluster(conn_matrix, opts.crossbarSize, base_quality);
                clusters = prune.clusters{i};
                prune.cmap{i} = cnn_cluster_createmap(conn_matrix, opts.utilth(i), clusters);
                prune.unclustered_prev{i} = prune.unclustered_curr{i};
                [currentLayerWeights, prune] = cnn_cdp(currentLayerWeights, i, prune);
            % old cmap and updated pmap is used for pruning
            % re-cluster the layer if the change in pruned connections is
            % less than tolerance
            else
                fprintf(fid, 'FC Layer: %d \t unclustered current: %f \t unclustered previous: %f\n', i, prune.unclustered_curr{i}, prune.unclustered_prev{i}); 
                if (abs(prune.unclustered_curr{i} - prune.unclustered_prev{i}) >= tol)
                    fprintf(fid, 'Pruning continues...\n'); % for debug only
                    prune.unclustered_prev{i} = prune.unclustered_curr{i};
                    [currentLayerWeights, prune] = cnn_cdp(currentLayerWeights, i, prune);
                % try to cluster the currently unclustered synapses - Note
                % pruning of unclustered should be given more importance
                % than clustering
                else
                    fprintf(fid, 'Clustering of unclustered synapses being tried...\n'); % for debug only
                    % get the unclustered conn_matrix (conn_mat_uc)
                    conn_mat_uc = prune.pmap{i};
                    conn_mat_uc(prune.cmap{i} == 1) = 0;
                    % run clustering now on unclustered synapses
                    clusters = {};
                    clusters.size = 0;
                    base_quality = opts.cluster_base_quality_max + 0.1;
                    while (clusters.size == 0)
                        base_quality = base_quality - 0.1;
                        if (base_quality <= opts.cluster_base_quality_min)
                            break; 
                        end
                        fprintf(fid, 'Base_quality tried for clustering: %f\n', base_quality);
                        clusters = cnn_size_constrained_cluster(conn_mat_uc, opts.crossbarSize, base_quality);
                    end
                    
                    % augment the new found clusters for ith layer to older ones
                    if (clusters.size ~= 0)
                        prune = cnn_add_clusters(prune, i, clusters);
                        % get the conn_matrix, clusters and then proceed with cdp
                        conn_matrix = logical(prune.cmap{i}) | logical(prune.pmap{i}); % conn_matrix is union of pmap and cmap
                        clusters = prune.clusters{i};
                        prune.cmap{i} = cnn_cluster_createmap(conn_matrix, opts.utilth(i), clusters); % cmap gets updated with new clusters
                    end
                    
                    prune.unclustered_prev{i} = prune.unclustered_curr{i};
                    fprintf(fid, 'Pruning continues...\n'); % for debug only
                    [currentLayerWeights, prune] = cnn_cdp(currentLayerWeights, i, prune);
                end
            end
               
        % accuracy dependent pruning only
        elseif (opts.prunemode == 1)
            [currentLayerWeights, prune] = cnn_adp(currentLayerWeights, i, prune);
        end
        cnn(opts.fcIndices(i)).Weights = currentLayerWeights;
    end
end