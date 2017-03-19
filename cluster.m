function [clusters, status] = cluster(conn_matrix, base_quality)
%CLUSTER runs spectral clustering using input-output pairs on a
%conenctivity matrix
%status: 1-returns with kmeans success, 0-kmeans returned with kmeans
%failure

    global DEBUG;

    conn_matrix = double(conn_matrix);
    %base_quality = sum(sum(conn_matrix))/(size(conn_matrix,1)*size(conn_matrix,2)); % - user-defined threshold
    clusters = {};
    status = 1;
    
    % Setup the clustering problem
    num_clusters = 5;
    clusters.size = 0;
    
    if DEBUG, disp('Clustering a conn_matrix....'); end
    iter  = 0; % This iteration ensures all synapses get covered in the clustering process.
    remaining_conn = 1;
    while (sum(sum(conn_matrix)) ~= 0)
        % An iteration of clustering begins here
        iter = iter + 1;
        remaining_conn_nxt = sum(sum(conn_matrix))/(size(conn_matrix,1)*size(conn_matrix,2));
        if ((remaining_conn_nxt == remaining_conn) && (remaining_conn_nxt ~= 1)) % stopping condition - 4 decimal precision equality testing
            break;
        end
        remaining_conn = remaining_conn_nxt;
        if DEBUG, disp(['Current Iteration:-----' num2str(iter) ...
                        '----remaining connections to cluster:----' num2str(remaining_conn)]); end
   
        % Create similarity matrix from connectivity matrix - first set of
        % values (row/col) are from input layer
        num_in = size(conn_matrix,2);
        num_out = size(conn_matrix,1);
        sim_matrix = zeros(num_in+num_out);
        sim_matrix(num_in+1:end, 1:num_in) = conn_matrix;
        sim_matrix(1:num_in, num_in+1:end) = conn_matrix';
        if DEBUG, fprintf('sim_matrix size: %d\n', size(sim_matrix,1)); end

        % Trim the sim_matrix - remove neurons(input as well as output) with no connections
        rows_nzero = find(sum(sim_matrix,2) ~= 0);
        rows_zero = find(sum(sim_matrix,2) == 0);
        num_out_nzero = find(rows_nzero>num_in,1);
        sim_matrix_nzero = sim_matrix(rows_nzero, rows_nzero);
        if DEBUG
            fprintf('sim_matrix size after trimming: %d\n', size(sim_matrix_nzero,1));
            fprintf('reduction fraction in clusering core due to trimming: %0.4f\n', 1-size(sim_matrix_nzero,1)/size(sim_matrix,1)); 
        end

        % Cluster based on spectral clustering algorithm - find clusters on
        % input and outputs which should be toegther
        degs = sum(sim_matrix_nzero, 2);
        D = sparse(1:size(sim_matrix_nzero, 1), 1:size(sim_matrix_nzero, 2), degs);
        L = D - sim_matrix_nzero;
        degs(degs == 0) = eps;
        D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
        L = D * L;
        diff   = 100*eps;
            
        % eigen value computation
        try 
            [U, ~] = eigs(L, num_clusters, diff); 
        catch
           %save('error/eigen_error.mat','L','num_clusters');
           %error ('Eigen Value computation failed!! ')
           status = 0;
           if DEBUG, fprintf('eigs failed ... saved the original cluster !!'); end
           break;
        end
        
        % k-means clustering -- PENDING!! - don't sub-cluster if kmeans
        % fails
        try
            C = kmeans(U, num_clusters, 'start', 'cluster', 'EmptyAction', 'singleton');
        catch
            status = 0;
            if DEBUG, fprintf('kmeans failed ... saved the original cluster !!'); end
            %save ('error/kmeans_error.mat','U','num_clusters');
            break;
        end

        %  Store the cluster and its metrics
        cluster_dist = zeros(num_clusters,1);
        if DEBUG, fprintf('Cluster_size \t num_in \t num_out \t   Quality \n'); end
        for j = 1:num_clusters
            cluster_dist(j) = sum(C==j);
            cluster_info = find(C==j);
            cluster_in = cluster_info(cluster_info < num_out_nzero);
            cluster_out = cluster_info(cluster_info >= num_out_nzero) - (num_out_nzero-1);
            num_inputs = size(cluster_in,1);
            num_outputs = size(cluster_out,1);
            cluster_quality = sum(sum(conn_matrix(cluster_out, cluster_in))) / ...
                              (num_inputs*num_outputs);
            if DEBUG, fprintf('\t%d\t\t\t\t%d\t\t\t%d\t\t\t%0.4f\n', cluster_dist(j), ...
                      num_inputs, num_outputs, cluster_quality); end
                  
            %store the found groups in clusters structure
            if (~isnan(cluster_quality) && cluster_quality > base_quality) %form a cluster only if it has input and output pair
                clusters.size = clusters.size + 1;
                clusters.C{clusters.size}.num_in = num_inputs;
                clusters.C{clusters.size}.num_out = num_outputs;
                clusters.C{clusters.size}.inputs = cluster_in;
                clusters.C{clusters.size}.outputs = cluster_out;
                clusters.C{clusters.size}.Q = cluster_quality;
                clusters.C{clusters.size}.map = conn_matrix(cluster_out, cluster_in);
                conn_matrix(cluster_out, cluster_in) = 0;
                %disp(sum(sum(conn_matrix))/(size(conn_matrix,1)*size(conn_matrix,2)));
            end
        end
        if DEBUG, fprintf('\n'); end
    end
    
end

