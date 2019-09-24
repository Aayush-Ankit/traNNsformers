%[er, bad] = cnntest_varFClayer(net, x, y)
clearvars -except cnn opts 

load ('/home/min/a/tibrayev/traNNsformer/datasets/mnist_uint8.mat')

%  feedforward
train_x = double(reshape(train_x',28,28,60000)) / 255;
train_y = double(train_y');
test_x = double(reshape(test_x',28,28,10000)) / 255;
test_y = double(test_y');

test_x = test_x(:,:,(1:10000)); 
test_y = test_y(:,(1:10000));  

foo = cnnff_varFClayers(cnn, test_x);
[~, h] = max(foo.o);
[~, a] = max(test_y);
bad = find(h ~= a);

er = numel(bad) / size(test_y, 2);