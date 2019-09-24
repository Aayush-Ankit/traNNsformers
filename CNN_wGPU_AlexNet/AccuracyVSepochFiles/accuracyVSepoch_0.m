prunemode = 0
imdsTest  = imageDatastore('/data/tibrayev/imagenet2012_zeromeanNormalized/val', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');
for e = 1:80
    fprintf(['epoch:' num2str(e) '\n'])
    Directory = dir(sprintf('./checkpoints/prunemode%d/epoch%d/', prunemode, e));
    FileName = Directory(3).name
    try
        load(sprintf('./checkpoints/prunemode%d/epoch%d/%s',prunemode,e,FileName))
        predictedLabels = classify(net, imdsTest, ...
        'MiniBatchSize', 60, ...
        'ExecutionEnvironment', 'gpu');
        accuracy_prunemode0(e,:) = sum(predictedLabels == imdsTest.Labels)/numel(imdsTest.Labels)*100;
        accuracy_prunemode0(end,:)
    catch
        disp('no checkpoint file found!')
    end
    clear net
end
save([sprintf('accuracyVSepochs_prunemode%d', prunemode) '.mat'], 'accuracy_prunemode0') 