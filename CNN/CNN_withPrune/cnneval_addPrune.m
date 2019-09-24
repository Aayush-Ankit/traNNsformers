function [loss] = cnneval_addPrune(net, loss, train_x, train_y, val_x, val_y)

net.testing = 1;
net = cnnff_addPrune(net, train_x, train_y);

loss.train.e(end+1) = net.L;

if nargin == 6
    net = cnnff_addPrune(net, val_x, val_y);
    loss.val.e(end+1) = net.L;
end

net.testing = 0;

end