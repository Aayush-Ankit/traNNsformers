function nn = cdp( nn, i )
%CDP cluster+accurcay driven pruning.
%   The prune map is the logical or of the pmap (accurcay map) and cmap
%   (cluster map).
% Inputs - neural network (nn), layer (i) 
    global fid;
    
    nn.pmap{i} = abs(nn.W{i}) >= nn.pruneth{i};
    nn.map{i} = logical(nn.cmap{i}) | logical(nn.pmap{i});
    nn.W{i} = nn.W{i} .* double(nn.map{i});
    prunestats = 100*sum(sum(nn.map{i}))/(size(nn.map{i},1) * size(nn.map{i},2));
    
    % compute the no. of unclustered synapses
    %unclus_map = nn.pmap{i};
    unclus_map = nn.map{i};
    unclus_map(nn.cmap{i} == 1) = 0;
    nn.unclustered_curr{i} = sum(sum(unclus_map))/(size(nn.map{i},1) * size(nn.map{i},2));
    
    fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);
end