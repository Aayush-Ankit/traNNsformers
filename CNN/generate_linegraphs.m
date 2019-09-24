clear; close all; clc
% This script generates the combined linegraphs (prunemode 2 & 3) 
% representing the fraction of unclustered synpases out of the remaining
% syanpses so as the visualize the benefits of online clustering over 
% offline clustering.
header;

%% Script to generate linegraph figures - pm2 and pm3 comparison of fractions of unclustered synapses
%addpath(genpath('/home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Matlab/'));

% pm3 plot data - extracted from the trace files - /home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Matlab/output/offline_clustering
lenet_mnist_pm3 = [0.55 0.71 0.01];

% pm2 plot data - extracted using analyze_cluster_mod from saved nns - /home/min/a/aankit/AA/AproxSNN-ControlledSparsity/Results/Algorithm
lenet_mnist_pm2 = [0.30 0.29 0.24];

% layer data - see nn of saved nn.mat files to see #layers being used
mnist_layers = [1 2 3];


% set the figure settings
%x0 = 0; y0 = 0; width = 11; height = 6; % for mnist
x0 = 0; y0 = 0; width = 8; height = 4; % for svhn
%x0 = 0; y0 = 0; width = 30; height = 12; % for cifar10

fig = figure('Units','inches',...
'position', [x0 y0 width height], ...
'PaperPositionMode','auto', 'PaperUnits', 'inches', 'PaperSize', [width height]);

set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','bold',...
        'FontSize', 14,...
        'FontName','Arial');
set(gca,'xtick',[])
hold on
box on

% plot the data
pm2 = lenet_mnist_pm2;
pm3 = lenet_mnist_pm3;

plot(pm2, '--ob', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b')
%axis([0.9 3.1 0 inf]) %mnist
axis([0.8 3.2 0 0.9])
grid on
hold on
plot(pm3, '--^k', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k')
%axis([0.9 3.1 0 inf]) %mnist
axis([0.8 3.2 0 0.9])
yticks([0 0.2 0.4 0.6 0.8])
grid on
hold on

ylabel('Unclustered Synapse Fraction', 'FontUnits','points', 'FontSize',18, 'FontName','Arial')
xlabel('Layers of Neural Network', 'FontUnits','points', 'FontSize',18, 'FontName','Arial')
leg = legend('TraNNsformer', 'Offline Clustering', 'FontUnits','points', 'FontName','Arial', 'Location','East');
%leg = legend('TraNNsformer', 'Offline Clustering', 'FontUnits','points', 'FontName','Arial', 'Position',[-0.5 0 0.5 0]);
set(leg, 'FontSize', 18)

% save the figure as pdf
tightfig;
% set with the print settings
print('CNNoutputs/mnist/lenet_mnist_pm23_lg','-dpdf');