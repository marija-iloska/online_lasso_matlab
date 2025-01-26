function [y_train, X_train, theta, y_test, X_test] = generate_data(N, N_test, P, var_features, var_theta,  num_zeros, var_y)

% Total data to create
N_total = N + N_test;

% Choose random indices to be 0s
j = datasample(1:P, num_zeros, 'replace', false);

% Generate random theta in the range between -rt, rt
theta = normrnd(0, var_theta, P, 1);

% Set chosen indices to 0s
theta(j) = 0;

% Create basis functions and data
X = zeros(N_total, P);

for n = 1:N_total
    X(n,:) = mvnrnd(zeros(1,P), var_features*eye(P));
end

% Generate linear model with Gaussian noise
y = X*theta;
y = y + mvnrnd(zeros(N_total,1), var_y*eye(N_total))';

% Split into training and test
y_train = y(1:N);
y_test = y(N+1:end);
X_train = X(1:N,:);
X_test = X(N+1:end,:);


end
