function [] = bar_plots(features, t0, T, p, K, formats)


[fsz, fszl, fszg, lwdt, color, grey, c_true, title_str] = formats{:};

pl = bar(t0:T, features', 1.0, 'stacked', 'FaceColor', 'flat', 'FaceAlpha', 1);
pl(1).CData = color;
pl(2).CData = grey;
hold on
yline(p, 'Color', c_true, 'LineWidth', lwdt-1)
ylim([0, K])
xlim([t0+1, T])
set(gca, 'FontSize',fszg)
legend('Correct', 'Incorrect', 'True Dim', 'FontSize', fszl)
title(title_str, 'FontSize', fsz)
ylabel(' Number of Features ', 'FontSize', fsz)
xlabel(' ', 'FontSize',fsz)
box on



end