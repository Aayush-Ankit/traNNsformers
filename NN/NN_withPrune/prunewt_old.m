function nn = prunewt(nn, current_epoch)
%PRUNEWT performs pruning based on prune mode and prune maps (accuracy_map
%and cluster_map)
%nn = prunewt(nn) returns a neural network strcuture with pruned weights
%based on the current prune threshold, crossbar utilization threshold & updates the two maps accordingly
    
    global fid;

    cstart = nn.clusterstartepoch;
    n = nn.n;
    for i = 1 : (n - 1)
        % accuracy + clustering dependent pruning
        if (nn.prunemode == 2)
            % no cluster map
            if (current_epoch < cstart)
                nn.pmap{i} = abs(nn.W{i}) >= nn.pruneth{i};
                nn.W{i} = nn.W{i} .* nn.pmap{i};
                prunestats = 100*sum(sum(nn.pmap{i}))/(size(nn.pmap{i},1) * size(nn.pmap{i},2));
                fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);
            % create the cluster map
            elseif (current_epoch == cstart)
                conn_matrix = nn.pmap{i};
                %base_quality = sum(sum(conn_matrix))/(size(conn_matrix,1)*size(conn_matrix,2));
                base_quality = 0.7;
                fprintf(fid, 'Base_quality before clustering: %f\n', base_quality);
                nn.clusters{i} = size_constrained_cluster(conn_matrix, nn.crossbarSize, base_quality);
                clusters = nn.clusters{i};
                nn.cmap{i} = cluster_createmap(conn_matrix, nn.utilth(i), clusters);
%                 nn.pmap{i} = abs(nn.W{i}) >= nn.pruneth{i};
%                 map = logical(nn.cmap{i}) | logical(nn.pmap{i});
%                 nn.W{i} = nn.W{i} .* double(map);
%                 prunestats = 100*sum(sum(map))/(size(map,1) * size(map,2));
%                 fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);
                cdp(nn, i);
            % use the previously created cluster map
            else
%                 nn.pmap{i} = abs(nn.W{i}) >= nn.pruneth{i};
%                 map = logical(nn.cmap{i}) | logical(nn.pmap{i});
%                 nn.W{i} = nn.W{i} .* double(map);
%                 prunestats = 100*sum(sum(map))/(size(map,1) * size(map,2));
%                 fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);
                cdp(nn, i);
            end
               
        % acc. dependent pruning only
        elseif (nn.prunemode == 1)
            nn.pmap{i} = abs(nn.W{i}) >= nn.pruneth{i};
            nn.W{i} = nn.W{i} .* nn.pmap{i};
            prunestats = 100*sum(sum(nn.pmap{i}))/(size(nn.pmap{i},1) * size(nn.pmap{i},2));
            fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);
        end % no pruning if prunemode = 0
    end
    
end