function [correct, incorrect, mse] = metrics(theta_est, idx_true, y_test, X_test)

% Feature calculation
idx = find(theta_est ~= 0)';
correct = sum(ismember(idx, idx_true));
incorrect = length(idx) - correct;

% MSE on Test Data
y_pred = X_test*theta_est;
mse = mean((y_test - y_pred).^2);



end