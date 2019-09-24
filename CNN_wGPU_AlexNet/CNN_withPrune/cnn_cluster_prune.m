function [cnn, prune_cp] = cnn_cluster_prune( cnn, opts, prune )

global fid;

%CLUSTER_PRUNE prunes previously formed synapse clusters based on a
%cluster_prune_threshold
    
% currently this function analyses the fractional distribution of the two
% types of synapses in the clusters in all the layers after training
% (i)   (cmap == 1) & (pmap == 1)
% (ii)  (cmap == 1) & (pmap == 0)
    
    fprintf(fid, 'Cluster prune ....\n');
   
    % for debug only
    for i = 1 : opts.numberFCLayers
        prunestats = 100* sum(sum(prune.map{i}))/(size(prune.map{i},1) * size(prune.map{i},2));
        fprintf(fid, 'FC Layer %d Pruned before cluster pruning : %2.2f\n', i, 100-prunestats);
    end
    
    prune_cp = prune;
    % Analyse the cluster scores of all invalid clusters
    for i = 1 : opts.numberFCLayers
        clusters = prune_cp.clusters{i};
        cmap = prune_cp.cmap{i};
        pmap = prune_cp.pmap{i};

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
        cluster_prune_th = prune_cp.cluster_prune_factor{i} * max(c_score);
        prune_cp.cluster_count{i} = 0;
        for j = 1:clusters.size
            if (c_score(j) ~= 0) % zero only if the Q value for cluster is zero
                prune_cp.cluster_count{i} = prune_cp.cluster_count{i} + 1; % only for debug
                inputs = clusters.C{j}.inputs;
                outputs = clusters.C{j}.outputs;
                if (c_score(j) <= cluster_prune_th)
                    clusters.C{j}.Q = 0; % making the cluster invalid (cluster is pruned)
                    temp_map1 = prune_cp.cmap{i};
                    temp_map2 = prune_cp.pmap{i};
                    temp_map1(outputs, inputs) = 0;
                    temp_map2(outputs, inputs) = 0;
                    prune_cp.cmap{i} = temp_map1;
                    prune_cp.pmap{i} = temp_map2;
                end
            end
        end
        prune_cp.clusters{i} = clusters;
        prune_cp.map{i} = logical(prune_cp.cmap{i}) | logical(prune_cp.pmap{i});
        cnn(opts.fcIndices(i)).Weights = (cnn(opts.fcIndices(i)).Weights) .* double(prune_cp.map{i});
    end

    for i = 1 : opts.numberFCLayers
        % for debug only
        prunestats = 100* sum(sum(prune_cp.map{i}))/(size(prune_cp.map{i},1) * size(prune_cp.map{i},2));
        fprintf(fid, 'Remaining clusters in FC Layer %d after cluster_pruning: %d \t pruned: %2.2f\n', i, prune_cp.cluster_count{i}, 100-prunestats); % only for debug
        if_hist = 0;
        cnn_analyse_cluster(prune_cp.clusters{i}, prune_cp.map{i}, if_hist);
    end
 
end

