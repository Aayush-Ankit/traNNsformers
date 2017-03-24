%% Runs different datasets - MNIST, SVHN and CIFAR-10 with their respective nn setups and records the results
clear ; close all; clc
warning('off','all')

dataset_path = ['/home/min/a/aankit/AA/ReSpArch-SpintronicSNNProcessor_DAC_2017/' ...
                'Matlab/spiking_relu_conversion-master/dlt_cnn_map_dropout_nobiasnn/data/%s_uint8.mat'];
            
dirspec = 'output/%s';
epochs_mnist = 40;

%%  MNIST
% original network - nn.size tells #mPEs used

% pruning only - nn.pmap tells #mPEs used
% clustured pruning - nn.cluster tells #mPEs used
data_name = 'mnist';
epochs = 40;
prune_slowdown = epochs_mnist / epochs;
net = [784, 1200, 1200, 10];
dataset_pathid = sprintf (dataset_path, data_name);
mkdir (sprintf (dirspec, data_name));

% run prunemode = 1 - pruning only
prunemode = 1;
run_fcn (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode)

% run prunemode = 2 - clustered pruning
prunemode = 2;
run_fcn (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode)

%% SVHN
data_name = 'svhn';
epochs = 80;
prune_slowdown = epochs_mnist / epochs;
net = [1024, 1200, 1200, 10];
dataset_pathid = sprintf (dataset_path, data_name);
mkdir (sprintf (dirspec, data_name));

% run prunemode = 1 - pruning only
prunemode = 1;
run_fcn (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode)

% run prunemode = 2 - clustered pruning
prunemode = 2;
run_fcn (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode)

%% CIFAR10
data_name = 'cifar10';
epochs = 120;
prune_slowdown = epochs_mnist / epochs;
net = [1024, 1200, 1200, 10];
dataset_pathid = sprintf (dataset_path, data_name);
mkdir (sprintf (dirspec, data_name));

% run prunemode = 1 - pruning only
prunemode = 1;
run_fcn (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode)

% run prunemode = 2 - clustered pruning
prunemode = 2;
run_fcn (data_name, dataset_pathid, net, epochs, prune_slowdown, prunemode)




