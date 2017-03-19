function [ clusters ] = subcluster( clusters_p, constraint, base_quality )
%SUBCLUSTER runs spectral clustering on previously formed
%clusters to form smaller clusters (sub-clusters)
    
    global DEBUG;

    % Setup the clustering problem
    clusters.size = 0;
    clusters.C = {};

    % traverse all the previously formed clusters to form further smaller
    % clusters
    for n = 1:clusters_p.size
        % read the old cluster
        num_in = clusters_p.C{n}.inputs;
        num_out = clusters_p.C{n}.outputs;
        inputs = clusters_p.C{n}.inputs;
        outputs = clusters_p.C{n}.outputs;
        quality = clusters_p.C{n}.Q;
        conn_matrix = clusters_p.C{n}.map;
        
        % Run connection matrix based cluster algo on the old cluster
        if DEBUG, disp(['Clustering previous clusters....' num2str(n) '....started....']); end
        if ((clusters_p.C{n}.num_in + clusters_p.C{n}.num_out) <= constraint) %store a cluster (with non-zero synapses) if it already satifies the constraint
            clusters.size = clusters.size + 1;
            clusters.C{clusters.size} = clusters_p.C{n};
            clusters.C{clusters.size}.status = 1;
            continue
        end
  
        [clusters_temp, status] = cluster(conn_matrix, base_quality);
        if DEBUG, disp(num2str(status)); end
        
        if (status == 0) %store a cluster (with non-zero synapses) as it is if cannot be subclustered
            clusters.size = clusters.size + 1;
            clusters.C{clusters.size} = clusters_p.C{n};
            clusters.C{clusters.size}.status = 0;
            continue
        end
        if DEBUG, disp('Clustering previous clusters....ended...'); end

        % Parse the old matrix inputs and outputs and combine with the new
        % cluster_temp
        off = clusters.size;
        for i = 1:clusters_temp.size
            clusters.C{off+i}.status = 1;
            clusters.C{off+i}.num_in = clusters_temp.C{i}.num_in;
            clusters.C{off+i}.num_out = clusters_temp.C{i}.num_out;
            clusters.C{off+i}.inputs = inputs(clusters_temp.C{i}.inputs); %update the input and output indices wrt original conn_matrix
            clusters.C{off+i}.outputs = outputs(clusters_temp.C{i}.outputs);
            clusters.C{off+i}.Q = clusters_temp.C{i}.Q;
            clusters.C{off+i}.map = clusters_temp.C{i}.map;
        end
        clusters.size = clusters.size + clusters_temp.size;
    end
    
end

