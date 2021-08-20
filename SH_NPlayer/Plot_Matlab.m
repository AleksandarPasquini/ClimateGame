
data = readtable('results.csv');


data = data(find(data.enhance_factor == 20),:);
data = data(find(data.pop_size == 500),:);
data = data(find(data.effective_threshold==15),:);
data = data(find(data.group_size == 25),:);


update_rule_list = ["random", "popular", "accumulated_best", ...
    "accumulated_worst", "accumulated_better", "current_best", "current_worst", ...
    "current_better", "fermi"];

legend_list = ["random", "popular", "accumulated\_best", ...
    "accumulated\_worst", "accumulated\_better", "current\_best", "current\_worst", ...
    "current\_better", "fermi"];

figure();
result = [];

Markers = {'+','o','*','x','v','d','^','s','>','<'};

for i = 1:length(update_rule_list)
    if i > 1
        hold on
    end
    data_rand_c = [];
    data_rand_t = [];
    data1 = data(find(data.pop_update_rule == update_rule_list(i)),:);
    for run = 0:49
        ind1 = find(data1.run_number == run);
        data_rand_c(:,run+1) = data1{ind1,10};
        data_rand_t(:,run+1) = data1{ind1,13};
    end
    result(:,i) = mean(data_rand_c,2);
    t = mean(data_rand_t,2);
    
    plot(t,result(:,i),strcat('-',Markers{i}),'LineWidth',2,'MarkerSize',4,'DisplayName',legend_list(i));
end

% legend({'y = sin(x)';'y = cos(x)'},'Location','southwest')
hold off
xlabel('time');
ylabel('proportion of cooperators');

legend('Location', 'Best', 'fontsize',12);

% legend('random', 'popular', 'accumulated best', 'accumulated\_worst', 'accumulated\_better', 'current\_best', 'current\_worst', 'current\_better', 'fermi')



% legend('DECC-G','DECC-D','DECC-DG','DECC-eDG','fontsize',12,'fontweight','bold');

