function nn = adp( nn, i )
%ADP accuracy-driven pruning.
%   The prune map the pmap (accurcay map) only.
% Inputs - nn - neural network, i - current layer being pruned.
    global fid;
    
    nn.pmap{i} = abs(nn.W{i}) >= nn.pruneth{i};
    nn.W{i} = nn.W{i} .* nn.pmap{i};
    prunestats = 100*sum(sum(nn.pmap{i}))/(size(nn.pmap{i},1) * size(nn.pmap{i},2));
    fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);

end

