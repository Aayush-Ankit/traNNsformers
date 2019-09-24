function stop = getMiniBatchLoss(info)

stop = false;

persistent MiniBatchLosses
global averageEpochLoss

if info.State == "start"
    MiniBatchLosses = 0;
    
elseif info.State == "iteration"
    MiniBatchLosses(end+1) = info.TrainingLoss;
    
elseif info.State == "done"
    MiniBatchLosses(end+1) = info.TrainingLoss;
    
    averageEpochLoss = sum(MiniBatchLosses(2:end))/numel(MiniBatchLosses(2:end));
    clear MiniBatchLosses
end
end