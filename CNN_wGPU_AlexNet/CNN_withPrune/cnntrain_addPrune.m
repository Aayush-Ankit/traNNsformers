function net = cnntrain_addPrune(net, x, y, opts, val_x, val_y)

global fid;



% NEW
    loss.train.e               = [];
    loss.train.e_frac          = [];
    loss.val.e                 = [];
    loss.val.e_frac            = [];
    opts.validation = 0;
    if nargin == 6
        opts.validation = 1;
    end

    
m = size(x, 3);
numbatches = m / opts.batchsize;
if rem(numbatches, 1) ~= 0
    error('numbatches not integer');
end
net.rL = [];


%new addition - added a variable to track the previous tr_err
tr_err_prev = 1;

for i = 1 : opts.numepochs
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    fprintf(fid, ['****************START of Epoch: ', num2str(i) '/' num2str(opts.numepochs) '****************\n']);


    TEpochSTART = tic;

    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
        
% NEW
        if (net.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>net.inputZeroMaskedFraction);
        end

        for lay = 2 : numel(net.layers)   %  for each layer
            if strcmp(net.layers{lay}.type, 'c')
                num_maps = net.layers{lay}.outputmaps;
                used_maps = rand(num_maps,1) > opts.dropout;
                net.layers{lay}.used_maps = used_maps;
            end
        end

        net = cnnff_addPrune(net, batch_x, batch_y);
        net = cnnbp_addPrune(net);
        net = cnnapplygrads_addPrune(net, opts);

        if isempty(net.rL)
            net.rL(1) = net.L;
        end
        net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;

    end



% NEW - train and validation errors
    val_x = val_x(:,:,(1:1000)); 
    val_y = val_y(:,(1:1000));
    if opts.validation == 1
        loss = cnneval_addPrune(net, loss, x, y, val_x, val_y);
        str_perf = sprintf('Full-batch train mse = %f, validation mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = cnneval_addPrune(net, loss, x, y);
        str_perf = sprintf('Full-batch train err = %f', loss.train.e(end));
    end
    
    opts.alpha = opts.alpha * net.scaling_learningRate;
    
    TEpochEND = toc(TEpochSTART);
    
% NEW addition - prune at the end of each epoch (uses both pmap and cmap) -
% removes discrete synapses
    if ((net.prunemode == 1) || (net.prunemode == 2))
        net = cnn_prunewt(net, i);
    end
% cluster_prune every epoch (after cluster_prune_start epoch)
    if ((net.prunemode == 2) && (i >= net.cluster_prune_start))
        net = cnn_cluster_prune(net);
    end
    
% NEW addition - increase the prune threshold of each layer after an epoch
% If the tr_error decreases then
% 1. increase the pruning threshold (discrete synapses)
% 2. inclrease the cluster_prune threshold (clustered synapses)
if ((net.prunemode == 1) || (net.prunemode == 2))
    if (tr_err_prev > loss.train.e(end))
        fprintf(fid, 'increasing the pruning threshold err_prev = %0.4f\t err_crr = %0.4f\n', tr_err_prev, loss.train.e(end));
        for p = net.firstFClayerIndex : net.n
            net.pruneth{p} = net.pruneth{p} + net.scaling_pruneRate(p);
            if ((net.prunemode == 2) && (i >= net.cluster_prune_start)) % start cluster_pruning towards the later end of training
                fprintf(fid, 'increasing the cluster pruning threshold\n');
                net.cluster_prune_factor{p} = net.cluster_prune_factor{p} + net.scale_clusterpruneRate(p);
            end
        end      
    end
    tr_err_prev = loss.train.e(end); 
end
    
    
    disp_msg1 = ['Epoch: ' num2str(i) '/' num2str(opts.numepochs) '-----time: ' num2str(TEpochEND) ' seconds.'];
    disp_msg2 = ['Mini-batch mean squared error on training set is ' num2str(mean(net.rL((end+1-numbatches):(end))))];
    disp_msg3 = [str_perf];
    fprintf(fid, '%s\n', disp_msg1);
    fprintf(fid, '%s\n', disp_msg2);
    fprintf(fid, '%s\n', disp_msg3);
    fprintf(['Epoch: ' num2str(i) '/' num2str(opts.numepochs) '-----time: ' num2str(TEpochEND) ' seconds\n']);
    [er, ~] = cnntest_addPrune(net, val_x(:,:,(1:1000)), val_y(:,(1:1000)));
    fprintf(fid, ['Epoch 1k sample test accuracy: ' num2str((1-er)*100) '\n']);
    fprintf(['Epoch 1k sample test accuracy: ' num2str((1-er)*100) '\n']);
    
       
    
% %     fprintf(fid,['Epoch 1k sample test accuracy: ' num2str((1-er)*100) '\n']);
% %     disp(['Epoch 1k sample test accuracy: ' num2str((1-er)*100)])
end



for lay = 2 : numel(net.layers)   %  for each layer
    if strcmp(net.layers{lay}.type, 'c')
        num_maps = net.layers{lay}.outputmaps;
        used_maps = ones(num_maps,1);
        net.layers{lay}.used_maps = used_maps;
    end
end
    
end