clear all
close all
clc

% GENERATE SYNTHETIC DATA
% Settings
var_y = 0.5;            % Observation noise Variance
ps = 28;                 % Number of 0s in theta
K = 40;                 % Number of available features
var_features = 1;      % Range of input data H
var_theta = 2;         % Variance of theta
N = 1500;                 % Number of data points
p = K - ps;             % True model dimension

% Initial batch of data
n0 = K+1;

%Create data
[y, X, theta] = generate_data(N, K, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


%% RERUNs 
MaxIter = 1;
XTy = X(1:n0,:)'*y(1:n0);
eyeK = eye(K);

[THETA, STATS] = lasso(X(1:n0,:), y(1:n0), 'CV', min(10, n0));
theta_init = THETA(:,STATS.IndexMinMSE);
%theta_init = mvnrnd(zeros(1,K), eyeK)';
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

J_proposed = [];
J_lasso = [];

correct = [];
incorrect = [];
correct_las = [];
incorrect_las = [];

for n = n0+1 : N


    % Standard LASSO
    [THETA, STATS] = lasso(X(1:n,:), y(1:n), 'CV', min(10, n));
    theta_lasso(n,:) = THETA(:,STATS.IndexMinMSE);

    % Receive new data point X(n)

    % Update top
    gj = gj + X(n,:)'*y(n);

    % Update Denominators for each feature
    dj_old = dj;
    dj = dj + X(n,:).^2;

   lambda = sqrt(sum(dj_old)*var_y);
   %lambda = sqrt(sum(abs(X(n,:)))/var_y);
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

    [J_proposed(end+1), ~] = pred_error_lasso(y, X, n, n0, var_y, theta_est, 0);
    [J_lasso(end+1), ~] = pred_error_lasso(y, X, n, n0, var_y, theta_lasso(n,:)', 0);

    idx_prop = find(theta_est ~= 0)';
    correct(end+1) = sum(ismember(idx_prop, idx_h));
    incorrect(end+1) = length(idx_prop) - correct(end);

    idx_lasso = find(theta_lasso(n,:) ~= 0);
    correct_las(end+1) = sum(ismember(idx_lasso, idx_h));
    incorrect_las(end+1) = length(idx_lasso) - correct_las(end);

end


epsilon = 1e-5;


[theta_olin, idx_prop, J_olin, plot_stats] = olasso(y, X, n0, epsilon, var_y, find(theta~=0));


stats_prop = [correct; incorrect];
stats_lasso = [correct_las; incorrect_las];

[correct_olin, incorrect_olin] = plot_stats{:};
stats_olin = [correct_olin; incorrect_olin];

figure(1)
plot(lambda_store)
title('LAMBDA', 'FontSize', 20)

%%
Nsz = length(theta_store(:,1));
figure;
non_zeros = find(theta ~=0);
k = datasample(non_zeros, 3, 'replace', false);
for i = 1:3
    plot(theta(k(i))*ones(1,Nsz), 'k', 'LineWidth', 2)
    hold on
    plot(theta_store(:,k(i)), 'r', 'LineStyle','--', 'Linewidth',1)
    hold on
    plot(theta_lasso(:,k(i))*ones(1,Nsz), 'b', 'LineStyle','-.')
    hold on
    plot(theta_olin(:,k(i))*ones(1,Nsz), 'g', 'LineStyle','-');
end

figure;
idx_zeros = find(theta == 0);
yline(0, 'k', 'LineWidth',1)
hold on
for k = 1:3
    plot(theta_store(:,idx_zeros(k)), 'r', 'LineStyle','--', 'Linewidth',1);
    hold on
    plot(theta_lasso(:,idx_zeros(k))*ones(1,Nsz), 'b', 'LineStyle','-.');
    hold on
    plot(theta_olin(:,idx_zeros(k))*ones(1,Nsz), 'g', 'LineStyle','-');
end


figure;
hold on
plot(J_proposed, 'r', 'LineWidth',1)
plot(J_olin, 'g', 'Linewidth',1)
plot(J_lasso, 'b', 'Linewidth',1)
hold off


%%  BAR PLOTS

% Colors, FontSizes, Linewidths
load plot_settings.mat

fsz = 20;
fszl = 18;

% Time range to plot
time_plot = n0+1:N;



% BAR PLOTS SPECIFIC RUN =========================================
figure('Renderer', 'painters', 'Position', [200 300 1500 400])

% JPLS
subplot(1,3,1)
formats = {fsz, fszl, lwdt, c_tpls, c_inc, c_true, 'PROP'};
bar_plots(stats_prop, n0+1, N, p, K, formats)

% OLinLASSO
subplot(1,3,2)
formats = {fsz, fszl, lwdt, c_olin, c_inc, c_true, 'OLinLASSO'};
bar_plots(stats_lasso, n0+1, N, p, K, formats)

% LASSO
subplot(1,3,3)
formats = {fsz, fszl, lwdt, c_mcmc, c_inc, c_true, 'LASSO'};
bar_plots(stats_olin, n0+1, N, p, K, formats)


