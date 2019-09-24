%% Get hardware parameters for FULL CNN network (including conv layers)
clc; clear all; 
load('/home/min/a/tibrayev/RESEARCH/traNNsformer/traNNsformers/traNNsformer_CNN/CNNoutputs/mnist/trace_prunemode0_numlayers7_with3FClayers_numfilters12and64_learningRate0.70_xbarutilmin0.30.mat')

data_name = 'mnist';
global fid
fid = fopen(sprintf('CNNoutputs/%s/trace_mPEs_for_fullCNN.txt', data_name), 'w');


xbar_size = 4;
num_mpe = zeros(cnn.n,1);
num_synapses = zeros(cnn.n,1);
avg_util = zeros(cnn.n,1);

fprintf (fid, 'Hardware Parameters for original network\n');
% scan the connectivity matrix layer-wise for each layer
for i = 1 : (cnn.n)
    if cnn.layers{i}.type == 'c'
        for m = 1: numel(cnn.layers{i}.k)
            for n = 1: numel(cnn.layers{i}.k{m})
                num_mpe(i) = num_mpe(i) + ceil(size(cnn.layers{i}.k{m}{n}, 1)/xbar_size) * ceil(size(cnn.layers{i}.k{m}{n}, 2)/xbar_size);
                num_synapses(i) = num_synapses(i) + size(cnn.layers{i}.k{m}{n}, 1) * size(cnn.layers{i}.k{m}{n}, 2);
            end
        end
    elseif cnn.layers{i}.type == 'f'
        num_mpe(i) = ceil(size(cnn.layers{i}.W, 1)/xbar_size) * ceil(size(cnn.layers{i}.W, 2)/xbar_size);
        num_synapses(i) = size(cnn.layers{i}.W, 1) * size(cnn.layers{i}.W, 2);
    end
end
fprintf(fid, 'Total Number of mPEs needed after transformation for layer: %d\n', num_mpe);
fprintf(fid, 'Total Number of Synapses needed on each layer: %d\n', num_synapses);
