function stop = getMiniBatchLoss2(info)

stop = false;

persistent MiniBatchLosses2
global averageEpochLoss2

if info.State == "start"
    MiniBatchLosses2 = 0;
    
elseif info.State == "iteration"
    MiniBatchLosses2(end+1) = info.TrainingLoss;
    
elseif info.State == "done"
    MiniBatchLosses2(end+1) = info.TrainingLoss;
    
    averageEpochLoss2 = sum(MiniBatchLosses2(2:end))/numel(MiniBatchLosses2(2:end));
    clear MiniBatchLosses2
end
end