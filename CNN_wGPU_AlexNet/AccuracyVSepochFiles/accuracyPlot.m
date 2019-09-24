% plot
load('accuracyVSepochs_prunemode0.mat')
load('accuracyVSepochs_prunemode1.mat')
load('accuracyVSepochs_prunemode2.mat')
load('accuracyVSepochs_prunemode2_2.mat')
f = figure;
plot(accuracy_prunemode0, 'LineWidth',2)
hold on
plot(accuracy_prunemode1, 'LineWidth',2)
plot(accuracy_prunemode2, 'LineWidth',2)
plot(accuracy_prunemode2_2, 'LineWidth',2)
xlabel('Training effort (epochs)')
ylabel('Test accuracy (%)')
legend('Our trained AlexNet', 'Pruning only', 'TraNNsformer w/ aggressive pruning', 'TraNNsformer w/ non-aggressive pruning')
axis([0 120 5 inf])
f = tightfig(f);
%saveas(f, sprintf('accuracyVStrainingEffort.png'))