close all; clear; clc;
%% Load pruned neural network - connectivity matrix
load nn_mnist_99.1_pruned.mat;
%% Cluster layer-wise - input dependent SC only (SC - Spectral Clustering)
n = 1;%nn.n;
for i = 1:n
    % Create the "input-dependent" similarity matrix(S) from connectivity
    % map matrix(conn)
    conn = double(nn.pmap{i}');
    figure(1),spy(conn);
    
    % pre-process to remove all zero rows
    rows_zeros = find(sum(conn,2) ~= 0);
    conn = conn(rows_zeros,:);
    
    % similarity matrix
    S = conn * conn';
    S(1:size(S,1)+1:end) = 0; %make the diagonal elemnets 0
    figure(2),
    fig_temp = uint8(S/max(max(S))*255);
    %imshow(fig_temp)
    
    % calculate degree matrix
    degs = sum(S, 2);
    D    = sparse(1:size(S, 1), 1:size(S, 2), degs);
    % calculate the unnormalized laplacian matrix
    L = D - S;
    % avoid dividing by zero
    degs(degs == 0) = eps;
    % calculate inverse of D
    D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
    % calculate normalized Laplacian
    L = D * L;
    % compute the eigenvectors corresponding to the k smallest % eigenvalues
    k = 16;
    diff   = 100*eps;
    [U, D] = eigs(L, k, diff);
    
    % now use the k-means algorithm to cluster U row-wise
    % C will be a n-by-1 matrix containing the cluster number for
    % each data point
    C = kmeans(U, k, 'start', 'cluster', 'EmptyAction', 'singleton');
    
    % print the cluster distribution
    cluster_dist = zeros(k,1);
    for j = 1:k
        cluster_dist(j) = sum(C==j);
    end
    disp(cluster_dist)
    
    % arrange to form the conn based on new clusters
    conn_clustered = zeros(size(nn.pmap{i}',1), size(nn.pmap{i}',2));
    conn_temp = nn.pmap{1}';
    find_temp = [];
    start = 0;
    for j = 1:k
        start = start + size(find_temp,1);
        find_temp = find(C==j);
        conn_clustered(start+1:start+size(find_temp,1),:) = conn_temp(find_temp,:);
    end
    figure(3),
    subplot(1,2,1),spy(nn.pmap{i})
    subplot(1,2,2),spy(conn_clustered')
end