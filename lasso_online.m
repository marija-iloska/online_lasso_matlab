clear all
close all
clc

% GENERATE SYNTHETIC DATA
% Settings
var_y = 1;            % Observation noise Variance
ps = 20;                 % Number of 0s in theta
K = 30;                 % Number of available features
var_features = 0.5;      % Range of input data H
var_theta = 2;         % Variance of theta
N = 300;                 % Number of data points
p = K - ps;             % True model dimension

% Initial batch of data
n0 = K+1;

%Create data
[y, X, theta] = generate_data(N, K, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


% LASSO from scratch

[THETA, STATS] = lasso(X, y, 'CV', 10);
THETA = THETA(:,STATS.Index1SE);
lambda_standard = STATS.Lambda1SE;


MaxIter = 3;
D = inv(X(1:n0,:)'*X(1:n0,:));
XTy = X(1:n0,:)'*y(1:n0);
theta_est = D*XTy;
gj = XTy;
% Denominators for each feature
for j = 1:K
    dj(j) = (X(1:n0,j)'*X(1:n0,j));

    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:K, j);

    % Each top
    gj(j) = gj(j) - X(1:n0,j)'*( X(1:n0, all_but_j{j})*theta_est(all_but_j{j}));
end




eyeK = eye(K);
lambda = 2;
for n = n0+1 : N

    % Receive new data point X(n)
    g = D*X(n,:)';
    g_bot = X(n,:)*g + var_y;
    g = g/g_bot;
    D = (eyeK - g*X(n,:))*D;

    % Compute residual predictive error
    e = (y(n) - X(n,:)*theta_est)^2;

    % Compute theoretical pred error
    non_zeros = (theta_est ~=0);
    A = X(n,:)*D*sign(theta_est);
    E = var_y + var_y*X(n,non_zeros)*D(non_zeros,non_zeros)*X(n,non_zeros)' + lambda^2*A^2;


    % Evaluate lambda
    tol = 1;
    deltaE = e - E;
    if deltaE > tol
        lambda = sqrt(lambda^2 + deltaE*A^2);
    elseif deltaE < - tol
        lambda = sqrt(lambda^2 - deltaE*A^2);
    else
        lambda = lambda;
    end

    % Update top
    gj = gj + X(n,:)'*y(n);
    %XTy = XTy + X(n,:)'*y(n);

    % Update Denominators for each feature
    dj = dj + X(n,:).^2;

    for i = 1:MaxIter

        for j = 1:K

            % Data term
            gj(j) = gj(j) - X(n,j)*( X(n,all_but_j{j})*theta_est(all_but_j{j})); 
            term1 = gj(j)/dj(j);

            % Penalty term
            term2 = lambda/dj(j);

            % Update
            theta_est(j) = soft_threshold(term1, term2);
        end
    end



end

