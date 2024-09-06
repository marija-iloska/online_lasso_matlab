function [y_train, H_train, theta, y_test, H_test] = generate_data(N, N_test, dy, var_h, rt,  p_s, var_y)

N_total = N + N_test;
% Choose random indices to be 0s
j = datasample(1:dy, p_s, 'replace', false);

% Generate random theta in the range between -rt, rt
theta = normrnd(0, rt, dy, 1);
%unifrnd(-rt, rt, dy, 1);

% Set chosen indices to 0s
theta(j) = 0;

% Create basis functions and data
%H = sin(normrnd(0, var_h, N, dy));
H = normrnd(0, var_h, N_total, dy);

y = H*theta;
y = y + mvnrnd(zeros(N_total,1), var_y*eye(N_total))';

y_train = y(1:N);
y_test = y(N+1:end);
H_train = H(1:N,:);
H_test = H(N+1:end,:);


end
