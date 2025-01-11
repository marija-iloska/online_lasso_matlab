function [theta, xx, xy, lambda] = online_lasso(yn, Xn, xx, xy, theta, all_but_j, var_y, K)


% Update top
xy = xy + Xn'*yn;

% Update Denominators for each feature
xx = xx + (Xn.^2);

% Lambda
lambda = sqrt(sum(xx)*var_y);


 for j = 1:K
 
    % Data term
    xy(j) = xy(j) - Xn(j)*( Xn(all_but_j{j})*theta(all_but_j{j})); 

    theta(j) = sign(xy(j))*max(abs(xy(j)) - lambda, 0)/xx(j);
 end


end