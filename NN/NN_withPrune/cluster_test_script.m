%clear; close all; clc
%load('mnist_acc_97.8_adp.mat');
%crossbar_size = 64;

%% run sc on conn_matrix
% [clusters, ~] = cluster(conn_matrix);

%% run sc on previous clusters
% clusters_p = clusters;
% clusters = {};
% clusters = subcluster(clusters_p);

%% run constrained clustering
figure(2),
for i = 1:nn.n-1
    subplot(1,nn.n-1,i),
    %conn_matrix = nn.pmap{i};
    conn_matrix = logical(nn.cmap{i}) | logical(nn.pmap{i});
    %cluster = size_constrained_cluster(conn_matrix, crossbar_size);
    analyse_cluster(nn.clusters{i}, conn_matrix);
end

%% analyzed the formed clusters
% analyse_cluster (clusters, conn_matrix);
% hold on;
% analyse_cluster (clusters_final, conn_matrix);
