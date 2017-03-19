function [ clusters ] = size_constrained_cluster(conn_matrix, crossbar_size, base_quality, debug)
%SIZE_CONSTRAINED_CLUSTER runs subclustering to get clusters such their
%sizes are less than a certain value (crossbar size (rows+column))
    
    global fid;
    
    tic
    global DEBUG;
    DEBUG = (nargin == 4) && (debug == 1); 
    % setup the problem
    m = 2; % m*crossbar_size is the constaint
    constraint = m*crossbar_size;
    
    % run sc on the conn_matrix
    [clusters, ~] = cluster(conn_matrix, base_quality);
    
    % run sc recursively on subclusters untill the constraints are met
    constraint_met = 1;
    while (constraint_met ~= 0) 
        constraint_met = 0;
        % run sc on previous clusters
        clusters_p = clusters;
        clusters = {};
        clusters = subcluster(clusters_p, constraint, base_quality);
        %check to if constraint is met
        n = clusters.size;
        for i = 1:n
            c_size = clusters.C{i}.num_in + clusters.C{i}.num_out;
            if ((clusters.C{i}.status == 1) && (c_size > m*crossbar_size))
                constraint_met = constraint_met + 1;
            end
        end
    end
    % use analyze_cluster function to extract cluster statistics
    if_hist = 0;
    analyse_cluster (clusters, conn_matrix, if_hist);
    
    t = toc;
    %disp(['clustering takes ' num2str(t) ' seconds.']);
    fprintf(fid, 'clustering takes %0.2f seconds\n', t);

end

