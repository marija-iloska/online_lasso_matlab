clear all
close all
clc

% GENERATE SYNTHETIC DATA
% Settings
var_y = 1;            % Observation noise Variance
ps = 20;                 % Number of 0s in theta
K = 30;                 % Number of available features
var_features = 1;      % Range of input data H
var_theta = 2;         % Variance of theta
N = 300;                 % Number of data points
p = K - ps;             % True model dimension

% Initial batch of data
t0 = K+1;

%Create data
[y, X, theta] = generate_data(N, K, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


% LASSO from scratch

[THETA, STATS] = lasso(X, y, 'CV', 10);
THETA = THETA(:,STATS.Index1SE);
lambda_standard = STATS.Lambda1SE;



XTX = X'*X;
D = inv(XTX);
XTy = X'*y;

% Denominators for each feature
for j = 1:K
    dj{j} = (X(:,j)'*X(:,j));

    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:K, j);
end

%%
MaxIter = 300;
theta_est = zeros(K,1);
lambda = 100;
for i = 1:MaxIter

    for j = 1:K

        
        % Data term
        term1 = XTy(j) - X(:,j)'*(X(:,all_but_j{j})*theta_est(all_but_j{j}));
        term1 = term1/dj{j};

        % Penalty term
        term2 = lambda/dj{j};

        % Update 
        theta_est(j) = soft_threshold(term1, term2);
    end
end





