function [correct, incorrect, mse, fscore] = metrics(theta_est, theta, K, idx_true, y_test, X_test)

% Feature calculation
idx = find(theta_est ~= 0)';
correct = sum(ismember(idx, idx_true));
incorrect = length(idx) - correct;

% MSE on Test Data
y_pred = X_test*theta_est;
mse = mean((y_test - y_pred).^2);

% Fscore
theta_est = (theta_est ~= zeros(K,1));
theta = (theta ~= zeros(K,1));
precision = sum(theta_est.*theta)/sum(theta_est==1);
recall = sum(theta_est.*theta)/(sum(theta==1));
fscore = 2*(precision*recall/(precision+recall));



end