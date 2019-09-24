function [nn, L, accuracyVStrainingEffort]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

global fid;

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
accuracyVStrainingEffort = [];
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;

%new addition - added a variable to track the previous tr_err
tr_err_prev = 1;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    if opts.validation == 1 && opts.testAccuracyEveryEpoch ~= 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
%     disp_msg = ['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' ...
%           '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf];
%     fprintf(fid, '%s\n', disp_msg);

    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    t = toc;
    
    % new addition - prune at the end of each epoch (uses both pmap and
    % cmap) - removes discrete synapses
    if ((nn.prunemode == 1) || (nn.prunemode == 2))
        nn = prunewt(nn, i);
    end
    % cluster_prune every epoch (after cluster_prune_start epoch)
    if ((nn.prunemode == 2) && (i >= nn.cluster_prune_start))
        nn = cluster_prune(nn);
    end
    
    % new addition - increase the prune threhold of each layer after an epoch
    % If the tr_error decreases then
    % 1. increase the pruning threshold (discrete synapses)
    % 2. inclrease the cluster_prune threshold (clustered synapses)
    if ((nn.prunemode == 1) || (nn.prunemode == 2))
    if (tr_err_prev > loss.train.e(end))
        fprintf(fid, 'increasing the pruning threshold err_prev = %0.4f\t err_crr = %0.4f\n', tr_err_prev, loss.train.e(end));
        for p = 1:nn.n-1
            nn.pruneth{p} = nn.pruneth{p} + nn.scaling_pruneRate(p);
            if ((nn.prunemode == 2) && (i >= nn.cluster_prune_start)) % start cluster_pruning towards the later end of training
                fprintf(fid, 'increasing the cluster pruning threshold');
                nn.cluster_prune_factor{p} = nn.cluster_prune_factor{p} + nn.scale_clusterpruneRate(p);
            end
        end      
    end
    end
    tr_err_prev = loss.train.e(end);
        
    disp_msg = ['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' ...
          '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf];
%     fprintf(fid, '%s\n', disp_msg);
    fprintf(fid, '%s\n', disp_msg);
    fprintf ('epoch: %d\t time: %s\n', i, num2str(t));
    
    if opts.testAccuracyEveryEpoch == 1
        [er, bad] = nntest(nn, val_x, val_y);
        fprintf(fid, '*****Test Accuracy: %2.2f%%.*****\n', (1-er)*100);
        accuracyVStrainingEffort(end+1) = (1-er)*100;
    end

end
end

