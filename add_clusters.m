function nn = add_clusters( nn, i, clusters )
%ADD_CLUSTERS augments new clusters formed for layer i to the previous set
%of clusters
%   nn - neural network, i - layer, clusters - clusters to be added
    
    old_size = nn.clusters{i}.size;
    nn.clusters{i}.size = nn.clusters{i}.size + clusters.size; 
    for j = 1:clusters.size
        nn.clusters{i}.C{old_size+j} = clusters.C{j};
    end

end

