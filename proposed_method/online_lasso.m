function [theta, d, v, lambda] = online_lasso(yn, Xn, d, v, theta, all_but_j, var_y, P)


% Update top
v = v + Xn'*yn;

% Update Denominators for each feature
d = d + (Xn.^2);


% Lambda
top = sum(Xn.^2./d);
bottom = sum(Xn.^2/(d.^2));
lambda = sqrt(var_y*top/bottom);


 for j = 1:P
 
    % Data term
    v(j) = v(j) - Xn(j)*( Xn(all_but_j{j})*theta(all_but_j{j})); 

    theta(j) = sign(v(j))*max(abs(v(j)) - lambda, 0)/d(j);
 end


end