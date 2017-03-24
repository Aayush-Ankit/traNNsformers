function nn = cluster_prune( nn )

global fid;

%GROUP_PRUNE checks whether if an existing cluster can be pruned altogether
%or not based on cmap & pmap
    
% currently this function analyses the fractional distribution of the two
% types of synapses in the clusters in all the layers after training
% (i)   (cmap == 1) & (pmap == 1)
% (ii)  (cmap == 1) & (pmap == 0)

%    figure(1), % plot the histogram of fractions of all types of synapses across all the clusters
    for i = 1:(nn.n-1)
        clusters = nn.clusters{i};
        cmap = nn.cmap{i};
        pmap = nn.pmap{i};
        
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
        cluster_prune_th = nn.cluster_prune_factor * max(c_score);
        nn.cluster_count{i} = 0;
        for j = 1:clusters.size
            if (c_score(j) ~= 0) % zero only if the Q value for cluster is zero
                nn.cluster_count{i} = nn.cluster_count{i} + 1; % only for debug
                inputs = clusters.C{j}.inputs;
                outputs = clusters.C{j}.outputs;
                if (c_score(j) <= cluster_prune_th)
                    clusters.C{j}.Q = 0; % making the cluster invalid (cluster is pruned)
                    temp_map1 = nn.cmap{i};
                    temp_map2 = nn.pmap{i};
                    temp_map1(outputs, inputs) = 0;
                    temp_map2(outputs, inputs) = 0;
                    nn.cmap{i} = temp_map1;
                    nn.pmap{i} = temp_map2;
                end
            end
        end
        nn.clusters{i} = clusters;
        nn.map{i} = logical(nn.cmap{i}) | logical(nn.pmap{i});
        nn.W{i} = nn.W{i} .* double(nn.map{i});
        
        % for debug only
        prunestats = 100* sum(sum(nn.map{i}))/(size(nn.map{i},1) * size(nn.map{i},2));
        fprintf(fid, 'Remaining clusters in Layer %d during cluster_pruning: %d \t pruned: %2.2f\n', i, nn.cluster_count{i}, 100-prunestats); % only for debug
        if_hist = 0;
        analyse_cluster(nn.clusters{i}, nn.map{i}, if_hist);
    end

end

