clear; close all; clc
% This script generates the combined histograms (prunemode 2 & 3) so as the
% visualize the benefits of online clustering over offline clustering.

%% Script to generate histogram figures - pm2 and pm3 comparison
%addpath(genpath('/home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Matlab/'));
load('/home/min/a/tibrayev/RESEARCH/traNNsformer/traNNsformers/traNNsformer_CNN/CNNoutputs/mnist/qual_vec_data.mat')
pm2 = mnist_pm2;
pm3 = mnist_pm3;
n = 3;

% set the figure settings
%x0 = 0; y0 = 0; width = 20; height = 7; % for mnist
x0 = 0; y0 = 0; width = 32; height = 8; % for svhn
%x0 = 0; y0 = 0; width = 30; height = 12; % for cifar10

fig = figure('Units','inches',...
'position', [x0 y0 width height], ...
'PaperPositionMode','auto', 'PaperUnits', 'inches', 'PaperSize', [width height+2]);

numbins_vec_1 = [15, 10, 1]; % for mnist
numbins_vec_2 = [15, 15, 4]; % for mnist
%numbins_vec = [10, 20, 20, 5]; % for svhn
%numbins_vec = [10, 20, 20, 5]; % for cifar10

for i = 1:n
    subplot(1,n,i)
    %set the plot settings
    set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','bold',...
        'FontSize', 28,...
        'FontName','Arial');
    hold on
    box on
    numbins = numbins_vec_1(i);
    h1 = histogram(pm2{i+5}, numbins, 'Normalization', 'probability', 'FaceColor', 'black');
    %axis([0 1 0 0.5])
    hold on
    numbins = numbins_vec_2(i);
    h2 = histogram(pm3{i+5}, numbins, 'Normalization', 'probability', 'FaceColor', 'cyan');
    %axis([0 1 0 0.5])
    xlabel('Cluster Quality')
    ylabel('Fraction of Clusters')
    hold off
end

legend('TraNNsformer', 'Offline Clustering', 'Orientation', 'horizontal');

%tightfig;
% set with the print settings
print('CNNoutputs/mnist/mnist_pm23','-dpdf', '-fillpage');