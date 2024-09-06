function [coeff] = soft_threshold(coeff_init, penalty_term)

if coeff_init > penalty_term
    coeff = coeff_init - penalty_term;
elseif coeff_init < - penalty_term
    coeff = coeff_init + penalty_term;
else
    coeff = 0;
end



end