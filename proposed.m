clear all
close all
clc

% GENERATE SYNTHETIC DATA
% Settings
var_y = 1;            % Observation noise Variance
ps = 11;                 % Number of 0s in theta
K = 17;                 % Number of available features
var_features = 1;      % Range of input data H
var_theta = 2;         % Variance of theta
N = 500;                 % Number of data points
N_test = 300;
p = K - ps;             % True model dimension

% Initial batch of data
n0 = 5;

%Create data
[y, X, theta, y_test, X_test] = generate_data(N, N_test, K, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


%% RERUNs 
MaxIter = 1;
XTy = X(1:n0,:)'*y(1:n0);
eyeK = eye(K);

[THETA, STATS] = lasso(X(1:n0,:), y(1:n0), 'CV', min(10, n0));
%theta_init = THETA(:,STATS.IndexMinMSE);
theta_init = mvnrnd(zeros(1,K), 0.1*eyeK)';
theta_prop = theta_init;

xy = XTy;
% Denominators for each feature
for j = 1:K
    xx(j) = (X(1:n0,j)'*X(1:n0,j));

    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:K, j);

    % Each top
    xy(j) = xy(j) - X(1:n0,j)'*( X(1:n0, all_but_j{j})*theta_prop(all_but_j{j}));
end


correct = [];
incorrect = [];
correct_las = [];
incorrect_las = [];

mse_prop = [];
mse_lasso = [];

for n = n0+1 : N


    % Standard LASSO
    [THETA, STATS] = lasso(X(1:n,:), y(1:n), 'CV', min(10, n));
    theta_lasso(n,:) = THETA(:,STATS.IndexMinMSE);

    % Receive new data point X(n)
    [theta_prop, xx, xy] = online_predictive_lasso(y(n), X(n,:), xx, xy, theta_prop, all_but_j, var_y, K);
    theta_store(n,:) = theta_prop;

%     % Update top
%     gj = gj + X(n,:)'*y(n);
% 
%     % Update Denominators for each feature
%     dj_old = dj;
%     dj = dj + X(n,:).^2;
% 
%     lambda = sqrt(sum(dj_old)*var_y);
% 
%     lambda_store(n) = lambda;
%     
%     for i = 1:MaxIter
% 
%          for j = 1:K
% 
%             % Data term
%             gj(j) = gj(j) - X(n,j)*( X(n,all_but_j{j})*theta_est(all_but_j{j})); 
%             term1 = gj(j)/dj(j);
% 
%             % Penalty term
%             term2 = lambda/dj(j);
% 
%             % Update
%             theta_est(j) = soft_threshold(term1, term2);
%         end
%         theta_store(n,:) = theta_est;
%     end


    idx_prop = find(theta_prop ~= 0)';
    correct(end+1) = sum(ismember(idx_prop, idx_h));
    incorrect(end+1) = length(idx_prop) - correct(end);

    idx_lasso = find(theta_lasso(n,:) ~= 0);
    correct_las(end+1) = sum(ismember(idx_lasso, idx_h));
    incorrect_las(end+1) = length(idx_lasso) - correct_las(end);



    y_prop = X_test*theta_prop;
    y_lasso = X_test*theta_lasso(n,:)';
   
    
    mse_prop(end+1) = mean((y_test - y_prop).^2);
    mse_lasso(end+1) = mean((y_test - y_lasso).^2);
    

    

end


epsilon = 1e-3;


[theta_olin, idx_prop, J_olin, plot_stats, mse_olin, y_olin] = olasso(y, X, n0, epsilon, var_y, find(theta~=0), X_test, y_test);
theta_lasso_est = theta_lasso(end,:)';
theta_olin_est = theta_olin(end,:)';

stats_prop = [correct; incorrect];
stats_lasso = [correct_las; incorrect_las];

[correct_olin, incorrect_olin] = plot_stats{:};
stats_olin = [correct_olin; incorrect_olin];





% figure(1)
% plot(lambda_store)
% title('LAMBDA', 'FontSize', 20)

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
k = datasample(idx_zeros, 3, 'replace', false);
yline(0, 'k', 'LineWidth',1)
hold on
for i = 1:3
    plot(theta_store(:,k(i)), 'r', 'LineStyle','--', 'Linewidth',1);
    hold on
    plot(theta_lasso(:,k(i))*ones(1,Nsz), 'b', 'LineStyle','-.');
    hold on
    plot(theta_olin(:,k(i))*ones(1,Nsz), 'g', 'LineStyle','-');
end


% figure;
% hold on
% plot(J_proposed, 'r', 'LineWidth',1)
% plot(J_olin, 'g', 'Linewidth',1)
% plot(J_lasso, 'b', 'Linewidth',1)
% hold off

figure;
hold on
plot(mse_prop, 'r', 'LineWidth',1)
plot(mse_olin, 'g', 'Linewidth',1)
plot(mse_lasso, 'b', 'Linewidth',1)
hold off
title('MSE on Test Data', 'FontSize',15)


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
bar_plots(stats_olin, n0+1, N, p, K, formats)

% LASSO
subplot(1,3,3)
formats = {fsz, fszl, lwdt, c_mcmc, c_inc, c_true, 'LASSO'};
bar_plots(stats_lasso, n0+1, N, p, K, formats)


