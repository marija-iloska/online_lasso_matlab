function [theta, xx, xy] = online_predictive_lasso(yn, Xn, xx, xy, theta, all_but_j, var_y, K)

% Update top
xy = xy + Xn'*yn;

lambda = sqrt(sum(xx)*var_y);
%lambda = sum(sqrt(xx));

% Update Denominators for each feature
xx = xx + Xn.^2;

 for j = 1:K
 

    % Data term
    xy(j) = xy(j) - Xn(j)*( Xn(all_but_j{j})*theta(all_but_j{j})); 
    term1 = xy(j)/xx(j);

    % Penalty term
    term2 = lambda/xx(j);

    % Update
    theta(j) = soft_threshold(term1, term2);
 end


end