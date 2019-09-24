function num_mpe = cnn_map_unclustered( unclustered_map, xbar_size )
%MAP_UNCLUSTERED function maps the unclustered synapses left after
%clustered pruning
%   unclustered_map - input is the map for unclustered synapses
%   num_mpe = outputs the #required to map the unclustered synapses
%   Algorithm - pick on first come first serve basis
        
    % scan the unclustered map for a lyer
    num_mpe = 0;
    for j = 1:size(unclustered_map,1)
        for k = 1:size(unclustered_map,2)
            % map the synapses spanned by an xbar starting here (top-left part of the xbar)
            if (unclustered_map(j,k) == 1)
                num_mpe = num_mpe + 1;
                j_end = min(j+xbar_size-1, size(unclustered_map,1));
                k_end = min(k+xbar_size-1, size(unclustered_map,2));
                unclustered_map(j:j_end, k:k_end) = 0;
            end
        end
    end         

end

