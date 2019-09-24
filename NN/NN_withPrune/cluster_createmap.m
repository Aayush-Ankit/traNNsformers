function [ cluster_map ] = cluster_createmap(conn_matrix, util_th, clusters)
%CLUSTER_CREATEMAP creates a cluster map based on conn_matrix and formed
%clusters
% Current implementation - uses a crossbar utilization based threshold

    % setup
    n = clusters.size;
    [h, k] = size(conn_matrix);
    base_q = sum(sum(conn_matrix)) / (h*k);
    cluster_map = zeros(h,k); % '1' tells which synapse is a part of a cluster (Q>base)
    
    % scan through all clusters formed
    for i = 1:n
        %if (clusters.C{i}.Q >= util_th)
            cluster_map(clusters.C{i}.outputs, clusters.C{i}.inputs) = 1;
        %end
    end
end

