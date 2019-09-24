function [ num_mpe, avg_util, num_mpe_unclustered ] = get_hardware_params_cnn(cnn, prunemode)
%GET_HARDWARE_PARAMS reuturns an estimate of the number of mPEs being used
%by the current mapping
%   prunemode dictates which data structure with nn has network topology
%   mapping information
% num_mpe - number of mPEs used for mapping
% avg_util - average utilization per PE (crossbar)
% path - path where nn infor is stored
% prunemode - which mode was used (no pruning, pruning, clustered pruning)
    
    global fid;
    %load (path)
    %xbar_size = nn.crossbarSize;
    xbar_size = 4;
    
    %output layer-wise details
    num_mpe = zeros(cnn.n,1);
    num_mpe_unclustered = zeros(cnn.n,1);
    avg_util = zeros(cnn.n,1);
    
    switch prunemode
        % no pruning
        case 0
            fprintf (fid, 'Hardware Parameters for original network\n');
            % scan the connectivity matrix layer-wise for FC layers only
            for i = (cnn.firstFClayerIndex) : (cnn.n)
                num_mpe(i) = ceil(size(cnn.layers{i}.W, 1)/xbar_size) * ceil(size(cnn.layers{i}.W, 2)/xbar_size);
                avg_util(i) = 1;
            end
                        
        % pruning only - no clustering
        case 1
            fprintf (fid, 'Hardware Parameters for pruned network\n');
            % scan the pmap layer-wise
            for i = (cnn.firstFClayerIndex) : (cnn.n)
                pmap_t = cnn.pmap{i};
                % remove rows with all zero entries
                rows_nzero = sum(pmap_t,2) ~= 0;
                cols_nzero = sum(pmap_t,1) ~= 0;
                pmap_t_nzero = pmap_t(rows_nzero, cols_nzero);
                h = ceil(size(pmap_t_nzero,1)/xbar_size);
                k = ceil(size(pmap_t_nzero,2)/xbar_size);
                num_mpe(i) = h * k;
                % scan a layer's pmap at crossbar granualarity
                map = zeros(h,k);
                map((1:size(pmap_t_nzero,1)), (1:size(pmap_t_nzero,2))) = pmap_t_nzero;
                avg_util(i) = sum(sum(map)) / size(map,1) / size(map,2);
            end
            
        % clustered pruning - online clustering
        case 2
            fprintf (fid, 'Hardware Parameters for online clustered network\n');
            % scan the map layer-wise - counting crossbars for clustered
            % synapses only - update to add the unclustered synapses also
            for i = (cnn.firstFClayerIndex) : (cnn.n)
                cluster_list = cnn.clusters{i};
                for j = 1:cluster_list.size
                    % process valid clusters only
                    cluster = cluster_list.C{j};
                    if (cluster.Q ~= 0)
                        % may need to check for all zero rows/columns - not included now (CHECK!)
                        num_mpe(i) = num_mpe(i) + ceil(cluster.num_in/xbar_size) * ceil(cluster.num_out/xbar_size);
                    end
                end
                unclustered_map = (cnn.pmap{i} == 1) & (cnn.cmap{i} == 0);
                num_mpe_unclustered(i) = cnn_map_unclustered(unclustered_map, xbar_size);
                num_mpe(i) = num_mpe(i) + num_mpe_unclustered(i);
                % not important now - UPDATE LATER!
                avg_util(i) = 1;
            end 
    end

end

