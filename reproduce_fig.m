clear all
close all
clc

% Add the paths
addpath(genpath('util/'), genpath('baselines/'))

% GENERATE SYNTHETIC DATA

R = 100;

% Settings
var_y = 1;              % Observation noise Variance
num_zeros = 75;                % Number of 0s in theta
P = 100;                 % Number of available features
var_features = 1;       % Variance of input features X
var_theta = 3;          % Variance of theta
N = 10000;                % Number of training data points
N_test = 200;           % Number of test data points
p = P - num_zeros;             % True model dimension

% Initial batch of data
n0 = 50;


tic
parfor run = 1:R


    % Create data
    [y, X, theta, y_test, X_test] = generate_data(N, N_test, P, var_features, var_theta,  num_zeros, var_y);
    idx_h = find(theta ~= 0)';



    %% Stream data
    tic
    [mse, fs, mst, stats] = stream_data(y, X, var_y, n0,N, P, idx_h, theta, y_test, X_test);
    toc

    % Concatenate feature evals
    stats_prop_run(run,:,:) = stats{1};
    stats_olin_run(run,:,:) = stats{2};
    stats_lasso_run(run,:,:) = stats{3};
    stats_occd_run(run,:,:) = stats{4};

    mse_prop_run(run,:) = mse(1,:);
    mse_olin_run(run,:) = mse(2,:);
    mse_lasso_run(run,:) = mse(3,:);
    mse_occd_run(run,:) = mse(4,:);

    fs_prop_run(run,:) = fs(1,:);
    fs_olin_run(run,:) = fs(2,:);
    fs_lasso_run(run,:) = fs(3,:);
    fs_occd_run(run,:) = fs(4,:);

    mst_prop_run(run,:) = mst(1,:);
    mst_olin_run(run,:) = mst(2,:);
    mst_lasso_run(run,:) = mst(3,:);
    mst_occd_run(run,:) = mst(4,:);



end
toc

fs_prop = mean(fs_prop_run,1);
fs_lasso = mean(fs_lasso_run,1);
fs_olin = mean(fs_olin_run,1);
fs_occd = mean(fs_occd_run,1);

mse_prop = mean(mse_prop_run,1);
mse_lasso = mean(mse_lasso_run,1);
mse_olin = mean(mse_olin_run,1);
mse_occd = mean(mse_occd_run,1);

mst_prop = mean(mst_prop_run,1);
mst_lasso = mean(mst_lasso_run,1);
mst_olin = mean(mst_olin_run,1);
mst_occd = mean(mst_occd_run,1);

stats_prop = squeeze(mean(stats_prop_run,1));
stats_lasso = squeeze(mean(stats_lasso_run,1));
stats_olin = squeeze(mean(stats_olin_run,1));
stats_occd = squeeze(mean(stats_occd_run,1));



%save('results/update2_exp2.mat')

%% MSE plots

% Colors, FontSizes, Linewidths
load plot_settings.mat

% Time range to plot
time_plot = n0+1:N;


% PLOTS
% Figure size and position
% figure;
% set(gcf, 'Position', [100, 30, 400, 1200]);
% 
% % Tiled layout for tighter subplots
% tiledlayout(5, 1, 'TileSpacing', 'Compact', 'Padding', 'Compact')
% 
% 
% % First plot: bar plot Proposed Online LASSO
% nexttile
% formats = {fsz, fszl, fszg, lwdt, c_olasso, c_inc, c_true, 'Proposed Online LASSO'};
% bar_plots(stats_prop(:, time_plot), time_plot(1), time_plot(end), p, P, formats)
% 
% % Second plot: bar plot OLinLASSO
% nexttile
% formats = {fsz, fszl, fszg, lwdt, c_olin, c_inc, c_true, 'OLinLASSO'};
% bar_plots(stats_olin(:, time_plot), time_plot(1), time_plot(end), p, P, formats)
% hold on
% 
% % Third plot: bar plot OCCD
% nexttile
% formats = {fsz, fszl, fszg, lwdt, c_lasso, c_inc, c_true, 'OCCD-TWL'};
% bar_plots(stats_occd(:, time_plot),  time_plot(1), time_plot(end), p, P, formats)
% hold on
% 
% % Third plot: MSE on Test Data
% nexttile
% hold on
% plot(mse_lasso(time_plot), 'Color', 'k', 'LineWidth', lwd_ms-1)
% plot(mse_occd(time_plot), 'Color', c_lasso, 'LineWidth', lwd_ms-1)
% plot(mse_olin(time_plot), 'Color', c_olin, 'LineWidth', lwd_ms, 'LineStyle', '-.')
% plot(mse_prop(time_plot), 'Color', c_olasso, 'LineWidth', lwd_ms, 'LineStyle', '--')
% hold off
% ylim([0,7])
% set(gca, 'FontSize', fszg)
% ylabel('MSE on Test Data', 'FontSize', fsz)
% legend('LASSO','OCCD-TWL', 'OLinLASSO', 'Proposed Online LASSO', 'FontSize', fszl)
% 
% % Fourth plot: F-Score
% nexttile
% hold on
% plot(fs_lasso(time_plot), 'Color', 'k', 'LineWidth', lwd_ms-1)
% plot(fs_occd(time_plot), 'Color', c_lasso, 'LineWidth', lwd_ms-1)
% plot(fs_olin(time_plot), 'Color', c_olin, 'LineWidth', lwd_ms, 'LineStyle', '-.')
% plot(fs_prop(time_plot), 'Color', c_olasso, 'LineWidth',lwd_ms, 'LineStyle', '--')
% hold off
% ylim([0.5, 1])
% set(gca, 'FontSize', fszg)
% ylabel('F-Score', 'FontSize', fsz)
% xlabel('n^{th} data point arrival', 'FontSize', fsz)
% legend('LASSO', 'OCCD-TWL', 'OLinLASSO', 'Proposed Online LASSO', 'FontSize',  fszl)

%%

figure;
%set(gcf, 'Position', [100, 30, 400, 1200]);

% Tiled layout for tighter subplots
t = tiledlayout(2, 3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% First plot: bar plot Proposed Online LASSO
nexttile(1)
formats = {fsz, fszl, fszg, lwdt, c_olasso, c_inc, c_true, 'Proposed Online LASSO'};
bar_plots(stats_prop(:, time_plot), time_plot(1), time_plot(end), p, P, formats)

% Second plot: bar plot OLinLASSO
nexttile(2)
formats = {fsz, fszl, fszg, lwdt, c_olin, c_inc, c_true, 'OLinLASSO'};
bar_plots(stats_olin(:, time_plot), time_plot(1), time_plot(end), p, P, formats)
hold on

% Third plot: bar plot OCCD
nexttile(3)
formats = {fsz, fszl, fszg, lwdt, c_lasso, c_inc, c_true, 'OCCD-TWL'};
bar_plots(stats_occd(:, time_plot),  time_plot(1), time_plot(end), p, P, formats)
hold on

% Third plot: MSE on Test Data
nexttile(4)
hold on
plot(mse_lasso(time_plot), 'Color', 'k', 'LineWidth', lwd_ms-1)
plot(mse_occd(time_plot), 'Color', c_lasso, 'LineWidth', lwd_ms-1)
plot(mse_olin(time_plot), 'Color', c_olin, 'LineWidth', lwd_ms, 'LineStyle', '-.')
plot(mse_prop(time_plot), 'Color', c_olasso, 'LineWidth', lwd_ms, 'LineStyle', '--')
hold off
ylim([0,7])
set(gca, 'FontSize', fszg)
ylabel('MSE on Test Data', 'FontSize', fsz)
legend('LASSO','OCCD-TWL', 'OLinLASSO', 'Proposed Online LASSO', 'FontSize', fszl)

% Fourth plot: F-Score
nexttile(5, [1,1.5])
hold on
plot(fs_lasso(time_plot), 'Color', 'k', 'LineWidth', lwd_ms-1)
plot(fs_occd(time_plot), 'Color', c_lasso, 'LineWidth', lwd_ms-1)
plot(fs_olin(time_plot), 'Color', c_olin, 'LineWidth', lwd_ms, 'LineStyle', '-.')
plot(fs_prop(time_plot), 'Color', c_olasso, 'LineWidth',lwd_ms, 'LineStyle', '--')
hold off
ylim([0.5, 1])
set(gca, 'FontSize', fszg)
ylabel('F-Score', 'FontSize', fsz)
xlabel('n^{th} data point arrival', 'FontSize', fsz)
legend('LASSO', 'OCCD-TWL', 'OLinLASSO', 'Proposed Online LASSO', 'FontSize',  fszl)

