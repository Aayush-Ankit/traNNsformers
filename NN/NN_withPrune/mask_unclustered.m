clear; close all; clc;
addpath(genpath('/home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Matlab/'));
dataset_path = ['/home/min/a/aankit/AA/ReSpArch-SpintronicSNNProcessor_DAC_2017/' ...
                'Matlab/spiking_relu_conversion-master/dlt_cnn_map_dropout_nobiasnn/data/%s_uint8.mat'];

%% MNIST
data_name = 'mnist';
dataset_pathid = sprintf (dataset_path, data_name);

load (dataset_pathid);
test_x  = double(test_x)  / 255;
test_y  = double(test_y);
    
load ('output/mnist/nn_prunemode2.mat')

[er, ~] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy before masking the unclustered synapses: %2.2f%%.\n', (1-er)*100);

% obtain the unclustered map & mask the unclusttered syanpses until
% penultimate layer
for i = 1:(nn.n-2)
    nn.W{i} = nn.W{i} .* nn.cmap{i};
end

[er, ~] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy after masking the unclustered synapses: %2.2f%%.\n', (1-er)*100);

%% SVHN
data_name = 'svhn';
dataset_pathid = sprintf (dataset_path, data_name);

load (dataset_pathid);
test_x  = double(test_x)  / 255;
test_y  = double(test_y);
    
load ('output/svhn/nn_prunemode2.mat')

[er, ~] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy before masking the unclustered synapses: %2.2f%%.\n', (1-er)*100);

% obtain the unclustered map & mask the unclusttered syanpses until
% penultimate layer
for i = 1:(nn.n-2)
    nn.W{i} = nn.W{i} .* nn.cmap{i};
end

[er, ~] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy after masking the unclustered synapses: %2.2f%%.\n', (1-er)*100);

%% CIFAR-10
data_name = 'cifar10';
dataset_pathid = sprintf (dataset_path, data_name);

load (dataset_pathid);
test_x  = double(test_x)  / 255;
test_y  = double(test_y);
    
load ('output/cifar10/nn_prunemode2.mat')

[er, ~] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy before masking the unclustered synapses: %2.2f%%.\n', (1-er)*100);

% obtain the unclustered map & mask the unclusttered syanpses until
% penultimate layer
for i = 1:(nn.n-2)
    nn.W{i} = nn.W{i} .* nn.cmap{i};
end

[er, ~] = nntest(nn, test_x, test_y);
fprintf('Test Accuracy after masking the unclustered synapses: %2.2f%%.\n', (1-er)*100);

