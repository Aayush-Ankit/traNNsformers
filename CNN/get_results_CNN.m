%% Train CNN network on MNIST dataset
clear all; close all; clc

dataset_path = ['/home/min/a/tibrayev/traNNsformer/datasets/%s_uint8.mat'];


dirspec = 'CNNoutputs/%s';
epochs_mnist = 5;


%% MNIST
data_name = 'mnist';  
epochs = 5; 
prune_slowdown = epochs_mnist / epochs;
cnn = struct(); 
dataset_pathid = sprintf(dataset_path, data_name);
mkdir (sprintf (dirspec, data_name));

% % run prunemode = 0 - No pruning, just clear training of NN
% prunemode = 0;
% cnn = run_cnn_addPrune(data_name, dataset_pathid, cnn, epochs, prune_slowdown, prunemode);

% % run prunemode = 1 - pruning only
% prunemode = 1;
% cnn = run_cnn_addPrune(data_name, dataset_pathid, cnn, epochs, prune_slowdown, prunemode);

% run prunemode = 2 - pruning and clustering
prunemode = 2;
cnn = run_cnn_addPrune(data_name, dataset_pathid, cnn, epochs, prune_slowdown, prunemode);