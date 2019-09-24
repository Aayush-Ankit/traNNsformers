function [currentLayerWeights, prune] = cnn_adp(currentLayerWeights, i , prune)
%ADP accuracy-driven pruning.
%   The prune map the pmap (accurcay map) only.
% Inputs - cnn - convolutional neural network, i - current FC layer being pruned.
    global fid;
     
    prune.pmap{i} = abs(currentLayerWeights) >= prune.pruneth{i};
    currentLayerWeights = (currentLayerWeights) .* prune.pmap{i};
    prunestats = 100*sum(sum(prune.pmap{i}))/(size(prune.pmap{i},1) * size(prune.pmap{i},2));
    fprintf(fid, 'Pruned percentage of FC Layer %d : %2.2f%%.\n', i, 100-prunestats);

end

