clear all
close all
clc

% GENERATE SYNTHETIC DATA
% Settings
var_y = 0.1;            % Observation noise Variance
ps = 20;                 % Number of 0s in theta
K = 50;                 % Number of available features
var_features = 1;      % Range of input data H
var_theta = 2;         % Variance of theta
N = 1000;                 % Number of data points
p = K - ps;             % True model dimension

% Initial batch of data
n0 = 15;

%Create data
[y, X, theta] = generate_data(N, K, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


% LASSO from scratch
[THETA, STATS] = lasso(X, y, 'CV', 10);
THETA = THETA(:,STATS.Index1SE);
lambda_standard = STATS.Lambda1SE;


MaxIter = 1;
XTy = X(1:n0,:)'*y(1:n0);
eyeK = eye(K);
theta_init = mvnrnd(zeros(1,K), eyeK)';
theta_est = theta_init;

gj = XTy;
% Denominators for each feature
for j = 1:K
    dj(j) = (X(1:n0,j)'*X(1:n0,j));

    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:K, j);

    % Each top
    gj(j) = gj(j) - X(1:n0,j)'*( X(1:n0, all_but_j{j})*theta_est(all_but_j{j}));
end

for n = 2 : N

    % Receive new data point X(n)

    % Update top
    gj = gj + X(n,:)'*y(n);
    %XTy = XTy + X(n,:)'*y(n);

    % Update Denominators for each feature
    dj_old = dj;
    dj = dj + X(n,:).^2;

    lambda = sqrt( sum(dj_old)*var_y );
    lambda_store(n) = lambda;
    
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
        theta_store(n,:) = theta_est;
    end



end

Nsz = length(theta_store(:,1));

figure(1)
plot(lambda_store)
title('LAMBDA', 'FontSize', 20)

%%
figure;
non_zeros = find(theta_est ~=0);
k = datasample(1:K, 1);
plot(theta(k)*ones(1,Nsz), 'k', 'LineWidth', 2)
hold on
plot(theta_store(:,k), 'r', 'LineStyle','--', 'Linewidth',1)
hold on
plot(THETA(k)*ones(1,Nsz), 'b', 'LineStyle','-.')


