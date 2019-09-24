function cnn_cp = cnn_cluster_prune( cnn )

global fid;

%CLUSTER_PRUNE prunes previously formed synapse clusters based on a
%cluster_prune_threshold
    
% currently this function analyses the fractional distribution of the two
% types of synapses in the clusters in all the layers after training
% (i)   (cmap == 1) & (pmap == 1)
% (ii)  (cmap == 1) & (pmap == 0)
    
    fprintf(fid, 'Cluster prune ....\n');
   
    % for debug only
    for i = cnn.firstFClayerIndex : cnn.n
        prunestats = 100* sum(sum(cnn.map{i}))/(size(cnn.map{i},1) * size(cnn.map{i},2));
        fprintf(fid, 'Layer %d Pruned before cluster pruning : %2.2f\n', i, 100-prunestats);
    end
    
    cnn_cp = cnn;
    % Analyse the cluster scores of all invalid clusters
    for i = cnn_cp.firstFClayerIndex : cnn_cp.n
        clusters = cnn_cp.clusters{i};
        cmap = cnn_cp.cmap{i};
        pmap = cnn_cp.pmap{i};

        % fraction of the different types of synapses in the clusters in a
        % layer
        frac_syn_c1p1 = zeros(clusters.size,1);
        num_syn_c1p1 = zeros(clusters.size,1);
        c_score = zeros(clusters.size,1);
        for j = 1:clusters.size
            if (clusters.C{j}.Q ~= 0)
                inputs = clusters.C{j}.inputs;
                outputs = clusters.C{j}.outputs;
                temp_cmap = cmap(outputs, inputs);
                temp_pmap = pmap(outputs, inputs);

                frac_syn_c1p1(j) = sum(sum(temp_cmap .* temp_pmap)) / (size(temp_cmap,1)*size(temp_cmap,2));
                num_syn_c1p1(j) = frac_syn_c1p1(j) * size(temp_cmap,1) * size(temp_cmap,2);
                c_score(j) = (clusters.C{j}.Q) * num_syn_c1p1(j); 
            end
        end

        % prune the clusters based on cluster_prune_th
        cluster_prune_th = cnn_cp.cluster_prune_factor{i} * max(c_score);
        cnn_cp.cluster_count{i} = 0;
        for j = 1:clusters.size
            if (c_score(j) ~= 0) % zero only if the Q value for cluster is zero
                cnn_cp.cluster_count{i} = cnn_cp.cluster_count{i} + 1; % only for debug
                inputs = clusters.C{j}.inputs;
                outputs = clusters.C{j}.outputs;
                if (c_score(j) <= cluster_prune_th)
                    clusters.C{j}.Q = 0; % making the cluster invalid (cluster is pruned)
                    temp_map1 = cnn_cp.cmap{i};
                    temp_map2 = cnn_cp.pmap{i};
                    temp_map1(outputs, inputs) = 0;
                    temp_map2(outputs, inputs) = 0;
                    cnn_cp.cmap{i} = temp_map1;
                    cnn_cp.pmap{i} = temp_map2;
                end
            end
        end
        cnn_cp.clusters{i} = clusters;
        cnn_cp.map{i} = logical(cnn_cp.cmap{i}) | logical(cnn_cp.pmap{i});
        cnn_cp.layers{i}.W = (cnn_cp.layers{i}.W) .* double(cnn_cp.map{i});

        % for debug only
        prunestats = 100* sum(sum(cnn_cp.map{i}))/(size(cnn_cp.map{i},1) * size(cnn_cp.map{i},2));
        fprintf(fid, 'Remaining clusters in Layer %d after cluster_pruning: %d \t pruned: %2.2f\n', i, cnn_cp.cluster_count{i}, 100-prunestats); % only for debug
        if_hist = 0;
        cnn_analyse_cluster(cnn_cp.clusters{i}, cnn_cp.map{i}, if_hist);
    end
    
end

