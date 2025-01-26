function [mse, fs, mst, stats] = stream_data(y, X, var_y, n0, N, P, idx_nonzeros, theta, y_test, X_test)


%% OLinLASSO init    

    % Initial LASSO estimate
    [B, STATS] = lasso(X(1:n0,:), y(1:n0), 'CV', min(n0,10));
    theta_init = B(:, STATS.IndexMinMSE);
    clear B STATS

    % Initial batch start
    theta_olin = theta_init;

    % Initial batch olin temrs
    xy0 = X(1:n0,:)'*y(1:n0);
    xx0 = X(1:n0,:)'*X(1:n0,:);

    % Step Size
    step = 0.01*n0/max(real(eig(xx0)));

    % Tolerance
    epsilon = 1e-3;

    % Initialize terms
    xy_olin = zeros(P,1);
    xx_olin = zeros(P,P);


    %% PROPOSED METHOD Initialize
    v = zeros(1,P);
    d = zeros(1,P);
    theta_prop = zeros(P,1);

    %% OCCD Initialize
    rn = zeros(1,P);
    Rn = zeros(P,P);
    theta_occd = zeros(P,1);

    %% Indices variable
    for j = 1:P
       % Indexes of all elements except jth
       all_but_j{j} = setdiff(1:P, j);
    end

    %% Stream data
    %tic
    for n = 1 : N

        % Receive new data point Xn, yn
        Xn = X(n,:);
        yn = y(n);

        % Call proposed online predictive lasso
        [theta_prop, d, v] = online_lasso(yn, Xn, d, v, theta_prop, all_but_j, var_y, P);


        % Call online linearized lasso
        if n > n0
            [theta_olin,  xx_olin, xy_olin] = olin_lasso(yn, Xn, xy0, xx0, xy_olin, xx_olin, theta_olin, epsilon, step, n0, n, P);
        end

        % OCCD
        [theta_occd, rn, Rn] = occd(yn, Xn, rn, Rn, n, P, theta_occd, all_but_j, var_y);


        % Evaluate models
        [correct_prop(n), incorrect_prop(n), mse_prop(n), fs_prop(n),mst_prop(n)] = metrics(theta_prop, theta, P, idx_nonzeros, y_test, X_test);
        [correct_olin(n), incorrect_olin(n), mse_olin(n), fs_olin(n),mst_olin(n)] = metrics(theta_olin, theta, P, idx_nonzeros, y_test, X_test);
        [correct_occd(n), incorrect_occd(n), mse_occd(n), fs_occd(n), mst_occd(n)] = metrics(theta_occd, theta, P, idx_nonzeros, y_test, X_test);

        % Standard LASSO - uses all points UP to n OFFLINE
        if n == N
            [theta_lasso, STATS] = lasso(X(1:n,:), y(1:n), 'CV', min(10, n));
            theta_lasso = theta_lasso(:,STATS.Index1SE);
            [correct_lasso, incorrect_lasso, mse_lasso, fs_lasso, mst_lasso] = metrics(theta_lasso, theta, P, idx_nonzeros, y_test, X_test);
        end
    end

    % Keep offline LASSO as constant
    len = length(fs_prop);
    fs_lasso = fs_lasso*ones(1,len);
    mse_lasso = mse_lasso*ones(1,len);
    mst_lasso = mst_lasso*ones(1,len);
    correct_lasso = correct_lasso*ones(1, len);
    incorrect_lasso = incorrect_lasso*ones(1,len);


    % Concatenate feature evals
    stats_prop = [correct_prop; incorrect_prop];
    stats_lasso = [correct_lasso; incorrect_lasso];
    stats_olin = [correct_olin; incorrect_olin];
    stats_occd = [correct_occd; incorrect_occd];

    mse = [mse_prop; mse_olin; mse_lasso; mse_occd];
    fs = [fs_prop; fs_olin; fs_lasso; fs_occd];
    mst = [mst_prop; mst_olin; mst_lasso; mst_occd];
    stats = {stats_prop, stats_olin, stats_lasso, stats_occd};



end