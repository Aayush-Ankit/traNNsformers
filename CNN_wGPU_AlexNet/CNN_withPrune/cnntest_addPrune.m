function [er, bad] = cnntest_addPrune(net, x, y)
    %  feedforward
    net.testing = 1;
    net = cnnff_addPrune(net, x, y);
    net.testing = 0;
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
end