function nn = cluster_prune_wrapper( nn, train_x, train_y )
%CLUSTER_PRUNE_WRAPPER wraps around the cluster_prune script to keep doing
%cluster_prune with increasing threshold until accuracy starts dropping.

% Cluster_prune incrementally (after training finishes) to prune while constraining the accuracy loss
% - note this isn't accompanied by any weight training now.
    global fid;

    % for debug only
    for i = 1:nn.n-1
        prunestats = 100* sum(sum(nn.map{i}))/(size(nn.map{i},1) * size(nn.map{i},2));
        fprintf(fid, 'Layer %d Pruned before cluster pruning : %2.2f\n', i, 100-prunestats); % only for debug
    end

    [last_err, ~] = nntest(nn, train_x, train_y);
    test_acc_base = (1-last_err)*100;
    
%% cluster_prune - implementation - check the clusters and prune based on the criteria (util * #synapses < th) 
% for every iteration, nn_cp undergoes updates while nn has the original
% network (before the current iteration) which can be used to revert the
% changes.

    while (true)
        disp('Entering loop...\n')
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


        %% Evaluate the degradation in accuracy and revert back cluster_prune if needed
        [err, ~] = nntest(nn_cp, train_x, train_y);
        test_acc_curr = (1-err)*100;
           
        if (nn_cp.cluster_prune_factor > 1)
            fprintf(fid, 'breaking off the cluster_prune loop cluster_prune NA\n');
            break;
        elseif ((test_acc_base-test_acc_curr) >= nn_cp.cluster_prune_acc_loss)
            fprintf(fid, 'breaking off the cluster_prune loop due to accuracy degradation\n');
            break;
        end
        
        nn_cp.cluster_prune_factor = nn_cp.cluster_prune_factor + 0.01;
        nn = nn_cp;
    end
    
    % analyze the effect of cluster_prune
    [err, ~] = nntest(nn, train_x, train_y);
    test_acc = (1-err)*100;
    for i = 1:nn.n-1
        prunestats = 100* sum(sum(nn.map{i}))/(size(nn.map{i},1) * size(nn.map{i},2));
        fprintf(fid, 'Layer %d Pruned after cluster pruning : %2.2f\n', i, 100-prunestats); % only for debug
    end

    fprintf(fid, 'Accuracy on training set after this group prune: %2.2f%%.\n', test_acc);

end

