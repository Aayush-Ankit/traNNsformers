function [currentLayerWeights, prune] = cnn_cdp(currentLayerWeights, i, prune)
%CDP cluster+accurcay driven pruning.
%   The prune map is the logical or of the pmap (accurcay map) and cmap
%   (cluster map).
% Inputs - neural network (nn), layer (i) 
    global fid;
    
    prune.pmap{i} = abs(currentLayerWeights) >= prune.pruneth{i};
    prune.map{i} = logical(prune.cmap{i}) | logical(prune.pmap{i});
    currentLayerWeights = (currentLayerWeights) .* double(prune.map{i});
    prunestats = 100*sum(sum(prune.map{i}))/(size(prune.map{i},1) * size(prune.map{i},2));
    
    % compute the no. of unclustered synapses
    %unclus_map = nn.pmap{i};
    unclus_map = prune.map{i};
    unclus_map(prune.cmap{i} == 1) = 0;
    prune.unclustered_curr{i} = sum(sum(unclus_map))/(size(prune.map{i},1) * size(prune.map{i},2));
    
    fprintf(fid, 'Pruned percentage of FC Layer %d : %2.2f%%.\n', i, 100-prunestats);
end