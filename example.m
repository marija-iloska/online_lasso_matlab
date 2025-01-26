clear all
close all
clc

% Add the paths
addpath(genpath('util/'), genpath('proposed_method/'))

% GENERATE SYNTHETIC DATA
% Settings
var_y = 0.1;            % Observation noise Variance
ps = 10;                % Number of 0s in theta
P = 20;                 % Number of available features
var_features = 3;       % Variance of input features X
var_theta = 1;          % Variance of theta
N = 5000;               % Number of training data points
N_test = 300;           % Number of test data points
p = P - ps;             % True model dimension

% Initial batch of data
n0 = 0;

% Create data
% y - label
% X - feature matrix (Phi in paper)
[y, X, theta, y_test, X_test] = generate_data(N, N_test, P, var_features, var_theta,  ps, var_y);
idx_h = find(theta ~= 0)';


%% PROPOSED METHOD INITIALIZE

% Initial batch start
theta_prop = zeros(P,1);

% Define initial terms
v = zeros(P,1);
d = zeros(1,P);

% Denominators for each feature
for j = 1:P
    % Indexes of all elements except jth
    all_but_j{j} = setdiff(1:P, j);
end


%% Stream data
tic

for n = n0+1 : N

    % Receive new data point X(n,:), y(n)
    Xn = X(n,:);
    yn = y(n);

    % Call proposed method
    [theta_prop, d, v, lambda(n)] = online_lasso(yn, Xn, d, v, theta_prop, all_but_j, var_y, P);

    % Evaluate models
    [correct_prop(n-n0), incorrect_prop(n-n0), mse_prop(n-n0), fs_prop(n-n0)] = metrics(theta_prop, theta, P, idx_h, y_test, X_test);
 

end
toc

% Concatenate feature evals
stats_prop = [correct_prop; incorrect_prop];




%% PLOTS

% Fromatting and colors
load plot_settings.mat
fsz = 17;
fszl = 15;
fszg = 13;


% Create figure 
figure('Renderer', 'painters', 'Position', [200 300 1500 400])

% MSE on test data 
subplot(1,3,3)
plot(mse_prop, 'r', 'LineWidth',2)
set(gca, 'FontSize',fszg)
ylabel('MSE on Test Data', 'FontSize',fszl)
xlabel('n^{th} data point arrival', 'FontSize',fszl)

% F-Score
subplot(1,3,2)
plot(fs_prop, 'r', 'Linewidth',2)
set(gca, 'FontSize',fszg)
ylabel('F-Score', 'FontSize',fszl)
xlabel('n^{th} data point arrival', 'FontSize',fszl)
ylim([0,1])

% Bar plots
subplot(1,3,1)
formats = {fsz, fszl, fszg, lwdt, c_olasso, c_inc, c_true, ''};
bar_plots(stats_prop, n0+1, N, p, P, formats)

sgtitle('Proposed Online LASSO', 'FontSize', fsz)

