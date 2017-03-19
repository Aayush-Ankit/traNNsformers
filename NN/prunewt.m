function nn = prunewt(nn, current_epoch)
%PRUNEWT performs pruning based on prune mode and prune maps (accuracy_map
%and cluster_map)
%nn = prunewt(nn) returns a neural network strcuture with pruned weights
%based on the current prune threshold, crossbar utilization threshold & updates the two maps accordingly
    
    global fid;

    cstart = nn.clusterstartepoch;
    tol = nn.tol; %for debug - 0.01 (shud be)
    n = nn.n;
    
    for i = 1 : (n - 1)
        % accuracy + clustering dependent pruning
        if (nn.prunemode == 2)
            % no cluster map
            if (current_epoch < cstart)
                nn = adp(nn, i);
            % create the cluster map (for the first time) and use updated cmap and updated pmap for pruning
            elseif (current_epoch == cstart)
                conn_matrix = nn.pmap{i};
                base_quality = nn.cluster_base_quality_max;
                fprintf(fid, 'Base_quality for clustering: %f\n', base_quality);
                nn.clusters{i} = size_constrained_cluster(conn_matrix, nn.crossbarSize, base_quality);
                clusters = nn.clusters{i};
                nn.cmap{i} = cluster_createmap(conn_matrix, nn.utilth(i), clusters);
                nn.unclustered_prev{i} = nn.unclustered_curr{i};
                nn = cdp(nn, i);
            % old cmap and updated pmap is used for pruning
            % re-cluster the layer if the change in pruned connections is
            % less than tolerance
            else
                fprintf(fid, 'Layer: %d \t unclustered current: %f \t unclustered previous: %f\n', i, nn.unclustered_curr{i}, nn.unclustered_prev{i}); 
                if (abs(nn.unclustered_curr{i} - nn.unclustered_prev{i}) >= tol)
                    fprintf(fid, 'Pruning continues...\n'); % for debug only
                    nn.unclustered_prev{i} = nn.unclustered_curr{i};
                    nn = cdp(nn, i);
                % try to cluster the currently unclustered synapses - Note
                % pruning of unclustered should be given more importance
                % than clustering
                else
                    fprintf(fid, 'Clustering of unclustered synapses being tried...\n'); % for debug only
                    % get the unclustered conn_matrix (conn_mat_uc)
                    conn_mat_uc = nn.pmap{i};
                    conn_mat_uc(nn.cmap{i} == 1) = 0;
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
                    end
                    
                    % augment the new found clusters for ith layer to older ones
                    if (clusters.size ~= 0)
                        nn = add_clusters (nn, i, clusters);
                        % get the conn_matrix, clusters and then proceed with cdp
                        conn_matrix = logical(nn.cmap{i}) | logical(nn.pmap{i}); % conn_matrix is union of pmap and cmap
                        clusters = nn.clusters{i};
                        nn.cmap{i} = cluster_createmap(conn_matrix, nn.utilth(i), clusters); % cmap gets updated with new clusters
                    end
                    
                    nn.unclustered_prev{i} = nn.unclustered_curr{i};
                    nn = cdp(nn, i);
                end
            end
               
        % accuracy dependent pruning only
        elseif (nn.prunemode == 1)
            nn = adp(nn, i);
        end
    end
    
end