clear all
close all
clc

% GENERATE SYNTHETIC DATA
% Settings
var_y = 0.5;            % Observation noise Variance
ps = 10;                 % Number of 0s in theta
K = 25;                 % Number of available features
var_features = 1;      % Range of input data H
var_theta = 2;         % Variance of theta
N = 700;                 % Number of data points
p = K - ps;             % True model dimension

% Initial batch of data
n0 = 3;

%Create data
[y, X, theta] = generate_data(N, K, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


%% RERUNs 
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

for n = n0+1 : N

    % Standard LASSO
    % LASSO from scratch
    [THETA, STATS] = lasso(X(1:n,:), y(1:n), 'CV', min(10, n));
    theta_lasso(n,:) = THETA(:,STATS.IndexMinMSE);

    % Receive new data point X(n)

    % Update top
    gj = gj + X(n,:)'*y(n);

    % Update Denominators for each feature
    dj_old = dj;
    dj = dj + X(n,:).^2;

    %lambda = sqrt( sum(dj_old + dj_old.^2/(X(n,:).^2))*var_y );
    %lambda = sqrt( var_y*sum( dj_old.* ( dj_old./(X(n,:).^2) + 1) )  );
   lambda = sqrt(sum(dj_old)*var_y);
   %lambda = sqrt(sum(abs(X(n,:)))/var_y);
    %lambda = sum(abs(X(n,:)).^2)/var_y;
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
non_zeros = find(theta ~=0);
k = datasample(non_zeros, 1);
for k = 1:length(non_zeros)
    plot(theta(non_zeros(k))*ones(1,Nsz), 'k', 'LineWidth', 2)
    hold on
    plot(theta_store(:,non_zeros(k)), 'r', 'LineStyle','--', 'Linewidth',1)
    hold on
    plot(theta_lasso(:,non_zeros(k))*ones(1,Nsz), 'b', 'LineStyle','-.')
end

figure;
idx_zeros = find(theta == 0);
yline(0, 'k', 'LineWidth',1)
hold on
for k = 1:length(idx_zeros)
    plot(theta_store(:,idx_zeros(k)), 'r', 'LineStyle','--', 'Linewidth',1);
    hold on
    plot(theta_lasso(:,idx_zeros(k))*ones(1,Nsz), 'b', 'LineStyle','-.');
end




