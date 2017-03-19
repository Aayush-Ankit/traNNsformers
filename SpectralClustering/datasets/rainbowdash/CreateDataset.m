FileName = 'rainbow_ava_80.jpg';
%FileName = 'test.jpg';
k = 2;
Neighbors = 25;

fprintf('Loading Image\n');

Img = imread(fullfile(FileName));
[m, n, d] = size(Img);
Data = reshape(Img, 1, m * n, []);

if d >= 2
    Data = (squeeze(Data))';
end
Data = double(Data);

Data = normalizeData(Data);

%csvwrite('rainbowdash.nld', Data');

% Do Clustering
fprintf('Creating SimGraph\n');
%SimGraph = SimGraph_NearestNeighbors(Data, Neighbors, 1);
SimGraph = SimGraph_Epsilon(Data, 0.3);

fprintf('Clustering\n');
C = SpectralClustering(SimGraph, k, 2);

D = convertClusterVector(C);
S = reshape(D, n, m);

if k == 2
    map = [0 0 0; 1 1 1];
else
    map = colormap('lines');
end

imshow(S, map, 'Border', 'tight');

clear all;