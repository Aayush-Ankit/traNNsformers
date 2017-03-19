FileName = 'rainbow_ava.jpg';
%FileName = 'fb.jpg';
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


% Approximate by rounding
mult = 10^2;
rData = round(Data * mult) / mult;
[~, ind, order] = unique(rData', 'rows', 'R2012a');

Data = Data(:, ind);

% OLD
%[Data, ~, order] = unique(Data', 'rows', 'R2012a');
%Data = Data';
%=====

% Do Clustering
fprintf('Creating SimGraph\n');
SimGraph = SimGraph_NearestNeighbors(Data, Neighbors, 1);

fprintf('Clustering\n');
C = SpectralClustering(SimGraph, k, 2);

D = convertClusterVector(C);
D = D(order);
S = reshape(D, m, n);

if k == 2
    map = [0 0 0; 1 1 1];
else
    %map = colormap('lines');
    map = zeros(3, k);
    for ii = 1:k
        ind = find(D == ii, 1);
        map(:, ii) = rData(:, ind);
    end
    map = map';
end

set(gca, 'position', [0 0 1 1], 'units', 'normalized')
%set(0, 'DefaultFigureMenu', 'none');
imshow(S, map, 'Border', 'tight');
hold on;
axis off;
truesize;
hold off;

clear all;