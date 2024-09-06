clear all
close all
clc

% GENERATE SYNTHETIC DATA
% Settings
var_y = 1;              % Observation noise Variance
ps = 5;                 % Number of 0s in theta
P = 20;                 % Number of available features
var_features = 1;       % Variance of input features X
var_theta = 1;          % Variance of theta
N = 800;                % Number of training data points
N_test = 300;           % Number of test data points
p = P - ps;             % True model dimension

% Initial batch of data
n0 = 0;

% Create data
[y, X, theta, y_test, X_test] = generate_data(N, N_test, P, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


%% PROPOSED METHOD INITIALIZE

% Initial batch start
theta_init = zeros(P,1);
theta_prop = theta_init;

% Define initial terms
xy = zeros(P,1);
xx = zeros(1,P);

% Denominators for each feature
for j = 1:P
    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:P, j);
end


%% Stream data

for n = n0+1 : N

    % Receive new data point Xn, yn
    Xn = X(n,:);
    yn = y(n);

    % Call proposed online predictive lasso
    tic
    [theta_prop, xx, xy] = online_predictive_lasso(yn, Xn, xx, xy, theta_prop, all_but_j, var_y, P);
    time_prop(n-n0) = toc;
    theta_prop_store(n,:) = theta_prop;


    % Evaluate models
    [correct_prop(n-n0), incorrect_prop(n-n0), mse_prop(n-n0), fs_prop(n-n0)] = metrics(theta_prop, theta, P, idx_h, y_test, X_test);
 

end

% Concatenate feature evals
stats_prop = [correct_prop; incorrect_prop];




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
    str_k = join(['\theta_{', num2str(k(i)), '}']);
    ylabel(str_k, 'FontSize', 20)
    hold off
end
xlabel('n^{th} data point arrival', 'FontSize', 15)


% Plot coefficient convergences of 0 coeffs
idx_zeros = find(theta == 0);
k = datasample(idx_zeros, 3, 'replace', false);
for i = 1:I
    subplot(3,2,2*i)
    hold on
    yline(0, 'k', 'LineWidth',1)
    plot(theta_prop_store(:,k(i)), 'r', 'LineStyle','--', 'Linewidth',1);
    hold off
    str_k = join(['\theta_{', num2str(k(i)), '}']);
    ylabel(str_k, 'FontSize', 20)
end
sgtitle('Convergence of Coefficients', 'FontSize', 15)
xlabel('n^{th} data point arrival', 'FontSize', 15)

%% MSE plots

figure;
subplot(1,2,1)
hold on
plot(mse_prop, 'r', 'LineWidth',1)
hold off
ylabel('MSE on Test Data', 'FontSize',15)
xlabel('n^{th} data point arrival', 'FontSize',15)

subplot(1,2,2)
hold on
plot(fs_prop, 'r', 'Linewidth',1)
hold off
ylabel('F-Score', 'FontSize',15)
xlabel('n^{th} data point arrival', 'FontSize',15)

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
formats = {fsz, fszl, lwdt, c_tpls, c_inc, c_true, 'PROPOSED'};
bar_plots(stats_prop, n0+1, N, p, P, formats)


