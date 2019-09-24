function cnn = cnn_adp(cnn, i )
%ADP accuracy-driven pruning.
%   The prune map the pmap (accurcay map) only.
% Inputs - cnn - convolutional neural network, i - current FC layer being pruned.
    global fid;
    
    cnn.pmap{i} = abs(cnn.layers{i}.W) >= cnn.pruneth{i};
    cnn.layers{i}.W = (cnn.layers{i}.W) .* cnn.pmap{i};
    prunestats = 100*sum(sum(cnn.pmap{i}))/(size(cnn.pmap{i},1) * size(cnn.pmap{i},2));
    fprintf(fid, 'Pruned percentage of Layer %d : %2.2f%%.\n', i, 100-prunestats);

end

