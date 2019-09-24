clear all
clc
warning('off', 'all')

trainDir = '/data2/backup/imagenet2012/train/';
trainDestDir = '/data/tibrayev/imagenet2012_zeromeanNormalized/train/';

numberOfNonJPGfilesTrain = 0;

dCatTrain = dir(trainDir);
isubTrain = [dCatTrain(:).isdir];
nameFoldersTrain = {dCatTrain(isubTrain).name}';
nameFoldersTrain(ismember(nameFoldersTrain, {'.', '..'})) = [];

normalizeTrain = tic;
for categoryIndex = 1:numel(nameFoldersTrain)
    currentCatName = char(nameFoldersTrain(categoryIndex));
    dFilesTrain = dir(strcat(trainDir,currentCatName));
    nameFilesTrain = {dFilesTrain.name}';
    nameFilesTrain(ismember(nameFilesTrain, {'.', '..'})) = [];
    
    for imageIndex = 1:numel(nameFilesTrain)
        currentFileName = char(nameFilesTrain(imageIndex));
    
        try
            img = imread(strcat(trainDir, currentCatName, '/', currentFileName));
% 
%             % Some images may be grayscale. Replicate the image 3 times to
%             % create an RGB image.
%             if ismatrix(img)
%                 img = cat(3,img,img,img);
%             end
% 
%             img = double(img);
%             for idx = 1:3
%                 meanCh = mean2(img(:,:,idx));
%                 stdCh  = std2(img(:,:,idx));
%                 imgNorm(:,:,idx) = (img(:,:,idx) - meanCh)/stdCh;
%             end
%                     
% 
%             % Resize the image as required for the CNN.
%             imgOutNorm = imresize(imgNorm, [227 227]);
%             clear imgNorm
%             
%             mkdir(trainDestDir, currentCatName);
%             imwrite(imgOutNorm, strcat(trainDestDir, currentCatName, '/', currentFileName), 'jpg');

        catch
            numberOfNonJPGfilesTrain = numberOfNonJPGfilesTrain + 1;
        end
    end
    disp(['Normalization of training category ' num2str(categoryIndex) ' is done!\n'])
end
normalizeTrainTime = toc(normalizeTrain);
disp(['Time required is: ' num2str(normalizeTrainTime) ' seconds.\n'])

%%
valDir = '/data2/backup/imagenet2012/val/';
valDestDir = '/data/tibrayev/imagenet2012_zeromeanNormalized/val/';
numberOfNonJPGfilesVal = 0;

dCatVal = dir(valDir);
isubVal = [dCatVal(:).isdir];
nameFoldersVal = {dCatVal(isubVal).name}';
nameFoldersVal(ismember(nameFoldersVal, {'.', '..'})) = [];

normalizeVal = tic;
for categoryIndex = 1:numel(nameFoldersVal)
    currentCatName = char(nameFoldersVal(categoryIndex));
    dFilesVal = dir(strcat(valDir,currentCatName));
    nameFilesVal = {dFilesVal.name}';
    nameFilesVal(ismember(nameFilesVal, {'.', '..'})) = [];
    
    for imageIndex = 1:numel(nameFilesVal)
        currentFileName = char(nameFilesVal(imageIndex));
    
        try
            img = imread(strcat(valDir, currentCatName, '/', currentFileName));

            % Some images may be grayscale. Replicate the image 3 times to
            % create an RGB image.
            if ismatrix(img)
                img = cat(3,img,img,img);
            end

            img = double(img);
            for idx = 1:3
                meanCh = mean2(img(:,:,idx));
                stdCh  = std2(img(:,:,idx));
                imgNorm(:,:,idx) = (img(:,:,idx) - meanCh)/stdCh;
            end
        

            % Resize the image as required for the CNN.
            imgOutNorm = imresize(imgNorm, [227 227]);
            clear imgNorm
            
            mkdir(valDestDir, currentCatName);
            imwrite(imgOutNorm, strcat(valDestDir, currentCatName, '/', currentFileName), 'jpg');

        catch
            numberOfNonJPGfilesVal = numberOfNonJPGfilesVal + 1;
        end
    end
    disp(['Normalization of validation category ' num2str(categoryIndex) ' is done!\n'])
end
normalizeValTime = toc(normalizeVal);
disp(['Time required is: ' num2str(normalizeValTime) ' seconds.\n'])