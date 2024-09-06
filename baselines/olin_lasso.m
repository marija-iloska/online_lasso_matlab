function [theta, xx, xy] = olin_lasso(yn, Xn, xy0, xx0, xy, xx, theta, epsilon, step, n0, n, p)



xx = xx + Xn'*Xn;
xy = xy + Xn'*yn;  

% Current gradient
grad = (xx*theta - xy)/n;
grad_init = (xx0*theta - xy0)/n0;

% Gradient difference
phi = grad - grad_init;

% Penalty param
lambda = (log(p)/n)^0.5;

loss = Inf;
loss_store = [];
while loss > epsilon

    if length(loss_store) > 2000
        error(message('Cannot converge. Either decrease step size or increase tolerance.'))
    end

    % Store old theta
    theta_old = theta;

    % Update steps
    temp = theta - step*( phi + (xx0*theta - xy0)/n0 );
    theta = sign(temp).*max(0, abs(temp) - lambda*step);

    % Loss update
    loss = sum( (theta - theta_old).^2);
    loss_store = [loss_store, loss];
    
end

end