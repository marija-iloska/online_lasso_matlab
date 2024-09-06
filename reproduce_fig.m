clear all
close all
clc

% Add the paths
addpath(genpath('util/'), genpath('baselines/'))

% GENERATE SYNTHETIC DATA

% Settings
var_y = 0.1;              % Observation noise Variance
ps = 3;                % Number of 0s in theta
K = 7;                 % Number of available features
var_features = 1;       % Variance of input features X
var_theta = 1;          % Variance of theta
N = 100;                % Number of training data points
N_test = 300;           % Number of test data points
p = K - ps;             % True model dimension

% Initial batch of data
n0 = 5;

% Create data
[y, X, theta, y_test, X_test] = generate_data(N, N_test, K, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';



%% INITIAL LASSO ESTIMATE
[B, STATS] = lasso(X(1:n0,:), y(1:n0), 'CV', min(n0,10));
theta_init = B(:, STATS.IndexMinMSE);
clear B STATS

lambda0 = sqrt(var_y*sum(X.*X,'all')/(N*N) );
[B, STATS] = lasso(X, y, 'lambda',  lambda0 );
theta_init = B(:, end);

[B, STATS] = lasso(X, y, 'CV', 10);
theta_cv = B(:, STATS.Index1SE);
lambda_offline = STATS.Lambda1SE;

%% OLinLASSO init

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
xy_olin = zeros(K,1);
xx_olin = zeros(K,K);


%% PROPOSED METHOD INITIALIZE

% Initial batch start
theta_prop = theta_init;

% Define initial terms
xy = xy0;
xx = sum((X(1:n0,:).*X(1:n0,:)),1);

% Denominators for each feature
for j = 1:K

    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:K, j);

    % Each top
    xy(j) = xy(j) - X(1:n0,j)'*( X(1:n0, all_but_j{j})*theta_prop(all_but_j{j}));
end


%% Stream data

for n = n0+1 : N

    % Receive new data point Xn, yn
    Xn = X(n,:);
    yn = y(n);

    % Standard LASSO - uses all points UP to n OFFLINE
    tic
    [theta_lasso, STATS] = lasso(X(1:n,:), y(1:n), 'CV', min(10, n));
    time_lasso(n-n0) = toc;
    theta_lasso = theta_lasso(:,STATS.Index1SE);
    theta_lasso_store(n,:) = theta_lasso;
    clear STATS

    % Call proposed online predictive lasso
    tic
    [theta_prop, xx, xy] = online_lasso(yn, Xn, xx, xy, theta_prop, all_but_j, var_y, K);
    time_prop(n-n0) = toc;
    theta_prop_store(n,:) = theta_prop;

    % Call online linearized lasso
    tic
    [theta_olin, xx_olin, xy_olin] = olin_lasso(yn, Xn, xy0, xx0, xy_olin, xx_olin, theta_olin, epsilon, step, n0, n, K);
    time_olin(n-n0) = toc;
    theta_olin_store(n,:) = theta_olin;


    % Evaluate models
    [correct_prop(n-n0), incorrect_prop(n-n0), mse_prop(n-n0), fs_prop(n-n0)] = metrics(theta_prop, theta, K, idx_h, y_test, X_test);
    [correct_lasso(n-n0), incorrect_lasso(n-n0), mse_lasso(n-n0), fs_lasso(n-n0)] = metrics(theta_lasso, theta, K, idx_h, y_test, X_test);
    [correct_olin(n-n0), incorrect_olin(n-n0), mse_olin(n-n0), fs_olin(n-n0)] = metrics(theta_olin, theta, K, idx_h, y_test, X_test);


end

% Concatenate feature evals
stats_prop = [correct_prop; incorrect_prop];
stats_lasso = [correct_lasso; incorrect_lasso];
stats_olin = [correct_olin; incorrect_olin];



%% COEFFICIENT PLOTS
Nsz = length(theta_prop_store(:,1));

% Plot coefficient convergences of non-0 coeffs

% Number of plots
I = 3;
figure;
non_zeros = find(theta ~=0);
k = datasample(non_zeros, 3, 'replace', false);
for i = 1:I
    subplot(3,2,2*i-1)
    hold on
    plot(theta(k(i))*ones(1,Nsz), 'k', 'LineWidth', 1)
    plot(theta_prop_store(:,k(i)), 'r', 'LineStyle','--', 'Linewidth',1)
    plot(theta_lasso_store(:,k(i)), 'b', 'LineStyle','-.')
    plot(theta_olin_store(:,k(i)), 'g', 'LineStyle','-');
    str_k = join(['\theta_{', num2str(k(i)), '}']);
    ylabel(str_k, 'FontSize', 20)
    hold off
end
xlabel('n^{th} data point arrival', 'FontSize', 15)
legend('True', 'Proposed', 'LASSO', 'OLinLASSO', 'FontSize', 10)


% Plot coefficient convergences of 0 coeffs
idx_zeros = find(theta == 0);
k = datasample(idx_zeros, 3, 'replace', false);
for i = 1:I
    subplot(3,2,2*i)
    hold on
    yline(0, 'k', 'LineWidth',1)
    plot(theta_prop_store(:,k(i)), 'r', 'LineStyle','--', 'Linewidth',1);
    plot(theta_lasso_store(:,k(i)), 'b', 'LineStyle','-.');
    plot(theta_olin_store(:,k(i)), 'g', 'LineStyle','-');
    hold off
    str_k = join(['\theta_{', num2str(k(i)), '}']);
    ylabel(str_k, 'FontSize', 20)
end
sgtitle('Convergence of Coefficients', 'FontSize', 15)
xlabel('n^{th} data point arrival', 'FontSize', 15)

legend('True', 'Proposed', 'LASSO', 'OLinLASSO', 'FontSize', 10)

%% MSE plots

figure;
subplot(1,2,1)
hold on
plot(mse_prop, 'r', 'LineWidth',1)
plot(mse_olin, 'g', 'Linewidth',1)
plot(mse_lasso, 'b', 'Linewidth',1)
hold off
ylabel('MSE on Test Data', 'FontSize',15)
xlabel('n^{th} data point arrival', 'FontSize',15)
legend('Proposed', 'OLinLASSO', 'LASSO', 'FontSize', 10)

subplot(1,2,2)
hold on
plot(fs_lasso, 'b', 'LineWidth',1)
plot(fs_olin, 'g', 'Linewidth',1)
plot(fs_prop, 'r', 'Linewidth',1)
hold off
ylabel('F-Score', 'FontSize',15)
xlabel('n^{th} data point arrival', 'FontSize',15)
legend('LASSO', 'OLinLASSO', 'Proposed', 'FontSize', 10)

%%  BAR PLOTS

% Colors, FontSizes, Linewidths
load plot_settings.mat

fsz = 20;
fszl = 18;

% Time range to plot
time_plot = n0+1:N;



% BAR PLOTS SPECIFIC RUN =========================================
figure('Renderer', 'painters', 'Position', [200 300 1500 400])

% Online LASSO proposed
subplot(1,3,1)
formats = {fsz, fszl, lwdt, c_olasso, c_inc, c_true, 'PROPOSED'};
bar_plots(stats_prop, n0+1, N, p, K, formats)

% OLinLASSO
subplot(1,3,2)
formats = {fsz, fszl, lwdt, c_olin, c_inc, c_true, 'OLinLASSO'};
bar_plots(stats_olin, n0+1, N, p, K, formats)

% LASSO
subplot(1,3,3)
formats = {fsz, fszl, lwdt, c_lasso, c_inc, c_true, 'LASSO'};
bar_plots(stats_lasso, n0+1, N, p, K, formats)

clear all

