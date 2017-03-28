function nn_cp = cluster_prune( nn )

global fid;

%GROUP_PRUNE checks whether if an existing cluster can be pruned altogether
%or not based on cmap & pmap
    
% currently this function analyses the fractional distribution of the two
% types of synapses in the clusters in all the layers after training
% (i)   (cmap == 1) & (pmap == 1)
% (ii)  (cmap == 1) & (pmap == 0)

%    figure(1), % plot the histogram of fractions of all types of synapses across all the clusters
    nn_cp = nn;
    for i = 1:(nn_cp.n-1)
        clusters = nn_cp.clusters{i};
        cmap = nn_cp.cmap{i};
        pmap = nn_cp.pmap{i};
        
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
        
        % prune the map based on cluster_prune_th
        cluster_prune_th = nn_cp.cluster_prune_factor * max(c_score);
        nn_cp.cluster_count{i} = 0;
        for j = 1:clusters.size
            if (c_score(j) ~= 0) % zero only if the Q value for cluster is zero
                nn_cp.cluster_count{i} = nn_cp.cluster_count{i} + 1; % only for debug
                inputs = clusters.C{j}.inputs;
                outputs = clusters.C{j}.outputs;
                if (c_score(j) <= cluster_prune_th)
                    clusters.C{j}.Q = 0; % making the cluster invalid (cluster is pruned)
                    temp_map1 = nn_cp.cmap{i};
                    temp_map2 = nn_cp.pmap{i};
                    temp_map1(outputs, inputs) = 0;
                    temp_map2(outputs, inputs) = 0;
                    nn_cp.cmap{i} = temp_map1;
                    nn_cp.pmap{i} = temp_map2;
                end
            end
        end
        nn_cp.clusters{i} = clusters;
        nn_cp.map{i} = logical(nn_cp.cmap{i}) | logical(nn_cp.pmap{i});
        nn_cp.W{i} = nn_cp.W{i} .* double(nn_cp.map{i});
        
        % for debug only
        prunestats = 100* sum(sum(nn_cp.map{i}))/(size(nn_cp.map{i},1) * size(nn_cp.map{i},2));
        fprintf(fid, 'Remaining clusters in Layer %d during cluster_pruning: %d \t pruned: %2.2f\n', i, nn_cp.cluster_count{i}, 100-prunestats); % only for debug
        if_hist = 0;
        analyse_cluster(nn_cp.clusters{i}, nn_cp.map{i}, if_hist);
    end

end

