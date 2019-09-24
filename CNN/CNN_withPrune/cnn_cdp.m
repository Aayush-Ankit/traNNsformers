function cnn = cnn_cdp( cnn, i )
%CDP cluster+accurcay driven pruning.
%   The prune map is the logical or of the pmap (accurcay map) and cmap
%   (cluster map).
% Inputs - neural network (nn), layer (i) 
    global fid;
    
    cnn.pmap{i} = abs(cnn.layers{i}.W) >= cnn.pruneth{i};
    cnn.map{i} = logical(cnn.cmap{i}) | logical(cnn.pmap{i});
    cnn.layers{i}.W = (cnn.layers{i}.W) .* double(cnn.map{i});
    prunestats = 100*sum(sum(cnn.map{i}))/(size(cnn.map{i},1) * size(cnn.map{i},2));
    
    % compute the no. of unclustered synapses
    %unclus_map = nn.pmap{i};
    unclus_map = cnn.map{i};
    unclus_map(cnn.cmap{i} == 1) = 0;
    cnn.unclustered_curr{i} = sum(sum(unclus_map))/(size(cnn.map{i},1) * size(cnn.map{i},2));
    
    fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);
end