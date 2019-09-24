%% Runs different datasets - MNIST, SVHN and CIFAR-10 with their respective nn setups and records the results
clear ; close all; clc
warning('off','all')

dataset_path = ['/home/min/a/tibrayev/RESEARCH/traNNsformer/datasets/%s_uint8.mat'];
            
dirspec = 'output/softwarePerspective_AccuracyVsTrainingEffortAnalysis/%s';
epochs_mnist = 40;


testAccuracyEveryEpoch = 1;

% original network - nn.size tells #mPEs used
% pruning only - nn.pmap tells #mPEs used
% clustured pruning - nn.cluster tells #mPEs used

%%  MNIST
% data_name = 'mnist';
% epochs = 40;
% prune_slowdown = epochs_mnist / epochs;
% net = [784, 1200, 1200, 10];
% dataset_pathid = sprintf (dataset_path, data_name);
% mkdir (sprintf (dirspec, data_name));
% 
% % run prunemode = 1 - pruning only
% prunemode = 1;
% run_fcn_general (data_name, dataset_pathid, net, epochs, prune_slowdown*0.7, prunemode, testAccuracyEveryEpoch)

% % run prunemode = 2 - clustered pruning
% epochs = 60;
% prunemode = 2;
% run_fcn_general (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode, testAccuracyEveryEpoch)

%% SVHN
% data_name = 'svhn';
% epochs = 100;
% prune_slowdown = epochs_mnist / epochs;
% net = [1024, 1200, 1200, 1200, 10];
% dataset_pathid = sprintf (dataset_path, data_name);
% mkdir (sprintf (dirspec, data_name));

% % run prunemode = 1 - pruning only
% prunemode = 1;
% run_fcn_general (data_name, dataset_pathid, net, epochs, prune_slowdown*0.5, prunemode, testAccuracyEveryEpoch)
% 
% % run prunemode = 2 - clustered pruning
% epochs = 120;
% prune_slowdown = epochs_mnist / epochs;
% prunemode = 2;
% run_fcn_general (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode, testAccuracyEveryEpoch)

%% CIFAR10
% data_name = 'cifar10';
% epochs = 160;
% prune_slowdown = epochs_mnist / epochs;
% net = [1024, 1200, 1200, 1200, 10];
% dataset_pathid = sprintf (dataset_path, data_name);
% mkdir (sprintf (dirspec, data_name));

% % run prunemode = 1 - pruning only
% prunemode = 1;
% run_fcn_general (data_name, dataset_pathid, net, epochs, prune_slowdown*0.6, prunemode, testAccuracyEveryEpoch)

% % run prunemode = 2 - clustered pruning
% epochs = 240;
% prune_slowdown = epochs_mnist / epochs;
% prunemode = 2;
% run_fcn_general (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode, testAccuracyEveryEpoch)

%% Extract qual_vec results from the cluster analysis for online clustering to plot the combined histograms
% % Load nn
% data_name = 'svhn';
% % path where trained nns are stored
% path_id = '/home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Results/Algorithm/%s/nn_prunemode2.mat';
% nnpath_id = sprintf (path_id, data_name);
% load (nnpath_id);
% 
% qual_vec = {};
% if_hist = 1;
% % find the updated clustering statistics & plot them (histograms)
% fig = figure(1);
% for i = 1:nn.n-1
%     subplot(1,nn.n-1,i),
%     % final connectivity matrix - logic or of cmap and pmap
%     conn_matrix = logical(nn.cmap{i}) | logical(nn.pmap{i});
%     qual_vec{i} = analyse_cluster_mod(nn.clusters{i}, conn_matrix, if_hist);
% end
% svhn_pm2 = qual_vec;
% 
% % save the qual_vec metadata for combine histogram
% save('qual_vec_data.mat', 'svhn_pm2', '-append')
