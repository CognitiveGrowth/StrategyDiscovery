%discovering strategies for decision-making in the Mouselab paradigm

mdp=MouselabMDPPayne()
[S0,mdp]=mdp.newEpisode();

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

alpha0=0.005;
nr_episodes=1000;

parfor rep=1:100
    [w(:,rep),MSE(:,rep),returns(:,rep)]=...
        semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,alpha0);
end


avg_MSE=mean(MSE,2);

R_total=mean(returns,2);

bin_width=25;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE,sem_RMSE]=binnedAverage(sqrt(avg_MSE),bin_width);
[avg_R,sem_R]=binnedAverage(R_total,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R,sem_R,'g-o','LineWidth',2), hold on
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
%ylim([0,10])
xlim([0,nr_episodes+5])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE,sem_RMSE,'g-o','LineWidth',2), hold on
xlim([0,nr_episodes+5])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
%legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'const','#observations','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)',...%'remaining time',
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)','last click decision',... %'early decision',' %'observation'
    };

figure()
bar(median(w,2)),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Semi-Gradient SARSA-Q without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)

%save DiscoveringDecisionStrategiesSARSAQ-12-30-2016.mat

%% inspect solution
[R_total,problems,states,actions,indices]=inspectPolicy(mdp,feature_extractor,mean(w,2),0,100)

[mean(indices.nr_acquisitions),sem(indices.nr_acquisitions(:))]
[mean(indices.PTPROB),sem(indices.PTPROB(:))]
[mean(indices.var_attribute),sem(indices.var_attribute(:))]
[mean(indices.var_alternative),sem(indices.var_alternative(:))]
[mean(indices.pattern),sem(indices.pattern(:))]
[mean(indices.percent_optimal_EV),sem(indices.percent_optimal_EV(:))]

%% apply Bayesian SARSA-Q algorithm
mdp=MouselabMDPPayne()
[S0,mdp]=mdp.newEpisode();

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

nr_episodes=3000;

for rep=1:4
    [glm(rep),MSE(:,rep),returns(:,rep)]=...
        BayesianSARSAQ(mdp,feature_extractor,nr_episodes);
end

%Add another 1000 episodes
for rep=1:20
    [glm(rep),MSE(1001:2000,rep),returns(1001:2000,rep)]=...
        BayesianSARSAQ(mdp,feature_extractor,nr_episodes,glm(rep));
end
save LowCost1000Episodes

%... and another 1000 episodes
load LowCost1000Episodes
nr_episodes=1000;
for rep=1:20
    [glm(rep),MSE(2001:3000,rep),returns(2001:3000,rep)]=...
        BayesianSARSAQ(mdp,feature_extractor,nr_episodes,glm(rep));
end



bin_width=200;
for r=1:20
    [avg_returns(:,r),sem_avg_return(:,r)]=binnedAverage(returns(:,r),bin_width)
end
best_run=argmax(avg_returns(end,:));

avg_MSE=mean(MSE,2);

R_total=mean(returns,2);

nr_episodes=size(R_total,1);
bin_width=100;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE,sem_RMSE]=binnedAverage(sqrt(avg_MSE),bin_width);
[avg_R,sem_R]=binnedAverage(R_total,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R,sem_R,'g-o','LineWidth',2), hold on
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
%ylim([0,10])
xlim([0,nr_episodes+5])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE,sem_RMSE,'g-o','LineWidth',2), hold on
xlim([0,nr_episodes+5])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
%legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'const','#observations','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)','remaining time',...
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)','last click decision',...
    'observation' %'early decision',' %'observation'
    };

weights=[glm(:).mu_n];
figure()
bar(weights(:,best_run)),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Bayesian SARSA without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)

[R_total,problems,states,actions,indices]=inspectPolicy(mdp,feature_extractor,weights(:,best_run),0,100)

[mean(indices.nr_acquisitions),sem(indices.nr_acquisitions(:))]
[nanmean(indices.PTPROB),sem(indices.PTPROB(:))]
[mean(indices.var_attribute),sem(indices.var_attribute(:))]
[mean(indices.var_alternative),sem(indices.var_alternative(:))]
[nanmean(indices.pattern),sem(indices.pattern(:))]
[mean(indices.percent_optimal_EV),sem(indices.percent_optimal_EV(:))]

for e=1:100
    [[zeros(1,mdp.nr_gambles+1);squeeze(problems(:,:,e))],actions(:,:,e)]
    pause()
end

%save result_Jan-2-2017-morning

%% what if people believed they receive every payoff
mdp=MouselabMDPPayne()
mdp.time_cost_per_sec=7/3600;
[S0,mdp]=mdp.newEpisode();

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

nr_episodes=1000;

for rep=1:20
    [glm(rep),MSE(:,rep),returns(:,rep)]=...
        BayesianSARSAQ(mdp,feature_extractor,nr_episodes);
end

for rep=1:20
    [glm(rep),MSE(1001:2000,rep),returns(1001:2000,rep)]=...
        BayesianSARSAQ(mdp,feature_extractor,nr_episodes,glm(rep));
end

bin_width=200;
for r=1:20
    [avg_returns(:,r),sem_avg_return(:,r)]=binnedAverage(returns(:,r),bin_width)
end
best_run=argmax(avg_returns(end,:));

avg_MSE=mean(MSE,2);

R_total=mean(returns,2);

nr_episodes=size(R_total,1);
bin_width=100;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE,sem_RMSE]=binnedAverage(sqrt(avg_MSE),bin_width);
[avg_R,sem_R]=binnedAverage(R_total,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R,sem_R,'g-o','LineWidth',2), hold on
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
%ylim([0,10])
xlim([0,nr_episodes+5])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE,sem_RMSE,'g-o','LineWidth',2), hold on
xlim([0,nr_episodes+5])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
%legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'const','#observations','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)','remaining time',...
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)','last click decision',...
    'observation' %'early decision',' %'observation'
    };

weights=[glm(:).mu_n];
figure()
bar(weights(:,best_run)),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Bayesian SARSA without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)

low_range=mdp.setPayoffRange(0,0.25);
high_range=mdp.setPayoffRange(0,9.99);

[R_total_low_range,problems_low_range,states_low_range,actions_low_range,indices_low_range]=...
    inspectPolicy(low_range,feature_extractor,weights(:,best_run),0,100)

[R_total_high_range,problems_high_range,states_high_range,actions_high_range,indices_high_range]=...
    inspectPolicy(high_range,feature_extractor,weights(:,best_run),0,100)

[mean(indices.nr_acquisitions),sem(indices.nr_acquisitions(:))]
[nanmean(indices.PTPROB),sem(indices.PTPROB(:))]
[mean(indices.var_attribute),sem(indices.var_attribute(:))]
[mean(indices.var_alternative),sem(indices.var_alternative(:))]
[nanmean(indices.pattern),sem(indices.pattern(:))]
[mean(indices.percent_optimal_EV),sem(indices.percent_optimal_EV(:))]

for e=1:100
    [[zeros(1,mdp.nr_gambles+1);squeeze(problems(:,:,e))],actions(:,:,e)]
    pause()
    most_probable=argmax(problems(:,1,e));
    less_probable=setdiff(1:mdp.nr_outcomes,most_probable);
    others=actions(1+less_probable,2:end,e);
    consistent_with_TTB(e)=and(all(not(isnan(actions(1+most_probable,:,e)))),...
        all(isnan(others(:))))
    consistent_with_SAT_TTB(e)=and(and(not(all(isnan(actions(1+most_probable,:,e)))),...
        all(isnan(others(:)))),any(isnan(actions(1+most_probable,:,e))))
        
    inspected_payoff_indices=find(not(isnan(actions(2:end,:,e))));
    payoffs=problems(:,2:end,e);
    inspected_payoffs=payoffs(inspected_payoff_indices);
    max_inspected_payoff(e)=max(inspected_payoffs);
    max_probability(e)=max(problems(:,1,e));
    max_EV(e)=max(inspected_payoffs)*max_probability(e);
    
    if numel(inspected_payoffs)>1
        max_inspected_payoff_before_last_move(e)=max(inspected_payoffs(1:end-1))
        max_EV_before_last_move(e)=max_probability(e)*max_inspected_payoff_before_last_move(e);
    end
        
end

max(max_inspected_payoff_before_last_move(consistent_with_TTB))
max(max_EV_before_last_move(consistent_with_TTB))
max(max_probability(consistent_with_TTB))

min(max_EV(consistent_with_TTB))
min(max_inspected_payoff(consistent_with_TTB))

min(max_EV(consistent_with_SAT_TTB))
min(max_inspected_payoff(consistent_with_SAT_TTB)) %guaranteed to be better than average
%save result_Jan-2-2017-morning



%% learn optimal policy for low stakes problems
clear

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

experiment=MouselabExperiment();

low_stakes=experiment.setPayoffRange(0,0.25);
low_stakes.payoff_ranges=[0,0.25];
low_stakes.time_per_click=1; %seconds per click

nr_training_episodes=6000;
nr_reps=10;
parfor rep=1:nr_reps
    tic()
    [glm_low_stakes(rep),MSE_low_stakes(:,rep),returns_low_stakes(:,rep)]=...
        BayesianSARSAQ(low_stakes,feature_extractor,nr_training_episodes);
    disp(['Repetition ',int2str(rep),' took ',int2str(round(toc()/60)),' minutes.'])
end


bin_width=200;
for r=1:nr_reps
    [avg_returns_low_stakes(:,r),sem_avg_return_low_stakes(:,r)]=...
        binnedAverage(returns_low_stakes(:,r),bin_width)
end
best_run_low_stakes=argmax(avg_returns_low_stakes(end,:));

avg_MSE_low_stakes=mean(MSE_low_stakes,2);

R_total_low_stakes=mean(returns_low_stakes,2);

nr_episodes=size(R_total_low_stakes,1);
bin_width=100;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE_low_stakes,sem_RMSE_low_stakes]=binnedAverage(sqrt(avg_MSE_low_stakes),bin_width);
[avg_R_low_stakes,sem_R_low_stakes]=binnedAverage(R_total_low_stakes,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R_low_stakes,sem_R_low_stakes,'g-o','LineWidth',2), hold on
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
title('Semi-gradient SARSA (Q) in Mouselab Task, Low Stakes','FontSize',18)
%ylim([0,10])
xlim([0,nr_episodes+5])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE_low_stakes,sem_RMSE_low_stakes,'g-o','LineWidth',2), hold on
xlim([0,nr_episodes+5])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
%legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'const','#observations','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)',...%'remaining time',...
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)',...%'last click decision',...
    'observation' %'early decision',' %'observation'
    };

weights_low_stakes=[glm_low_stakes(:).mu_n];
figure()
bar(weights_low_stakes(:,best_run_low_stakes)),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Bayesian SARSA, Low Stakes, ',int2str(nr_episodes),' episodes'],'FontSize',18)

[R_total_low_range,problems_low_range,states_low_range,actions_low_range,indices_low_range,has_high_dispersion]=...
    inspectPolicy(low_stakes,feature_extractor,glm_low_stakes(best_run_low_stakes),0,100)


save(['optimal_policy_low_stakes_',date()])

%% optimal policy for high stakes problems
clear

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

experiment=MouselabExperiment();

high_stakes=experiment.setPayoffRange(0.01,9.99);
high_stakes.payoff_ranges=[0.01,9.99];
high_stakes.time_per_click=1;

nr_training_episodes=6000;
nr_reps=10;
parfor rep=1:nr_reps
    tic()
    [glm_high_stakes(rep),MSE_high_stakes(:,rep),returns_high_stakes(:,rep)]=...
        BayesianSARSAQ(high_stakes,feature_extractor,nr_training_episodes);
    disp(['Repetition ',int2str(rep),' took ',int2str(round(toc()/60)),' minutes.'])
end

bin_width=200;
for r=1:nr_reps
    [avg_returns_high_stakes(:,r),sem_avg_return_high_stakes(:,r)]=binnedAverage(returns_high_stakes(:,r),bin_width)
    [avg_RMSE_high_stakes(:,r),sem_RMSE_high_stakes(:,r)]=binnedAverage(sqrt(MSE_high_stakes(:,r)),bin_width);
end
best_run_high_stakes=argmax(avg_returns_high_stakes(end,:));


avg_MSE_high_stakes=mean(MSE_high_stakes,2);

R_total_high_stakes=mean(returns_high_stakes,2);

nr_episodes=size(R_total_high_stakes,1);
bin_width=100;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE_high_stakes,sem_RMSE_high_stakes]=binnedAverage(sqrt(avg_MSE_high_stakes),bin_width);
[avg_R_high_stakes,sem_R_high_stakes]=binnedAverage(R_total_high_stakes,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R_high_stakes,sem_R_high_stakes,'g-o','LineWidth',2), hold on
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
%ylim([0,10])
xlim([0,nr_episodes+5])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE_high_stakes,sem_RMSE_high_stakes,'g-o','LineWidth',2), hold on
xlim([0,nr_episodes+5])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
%legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'const','#observations','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)',...%'remaining time',...
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)',...%'last click decision',...
    'observation' %'early decision',' %'observation'
    };

weights_high_stakes=[glm_high_stakes(:).mu_n];
figure()
bar(weights_high_stakes(:,best_run_high_stakes)),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Bayesian SARSA without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)

[R_total_high_range,problems_high_range,states_high_range,actions_high_range,indices_high_range]=...
    inspectPolicy(high_stakes,feature_extractor,glm_high_stakes(best_run_high_stakes),0,100)

save(['optimal_policy_high_stakes_',date()])

%% fit cost per click
clear
%cost_values=(0.5:1:5.5)/3600;

cost_values=[0,1];

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

experiment=MouselabExperiment();

for c=1:numel(cost_values)
    cost_per_click=cost_values(c);
    high_stakes=experiment.setPayoffRange(0.01,9.99);
    high_stakes.payoff_ranges=[0.01,9.99];
    high_stakes.time_per_click=1;
    high_stakes.time_cost_per_sec=cost_per_click;
    
    low_stakes=experiment.setPayoffRange(0.00,0.25);
    low_stakes.payoff_ranges=[0.00,0.25];
    low_stakes.time_per_click=1;
    low_stakes.time_cost_per_sec=cost_per_click;
    
    
    nr_training_episodes=2000;
    nr_reps=1;
    
    sigma0=0.1;
    glm_fit_high=BayesianGLM(13,sigma0);
    glm_fit_low=BayesianGLM(13,sigma0);
    
    mu0=[0,0.5,0.5,-0.5,-0.5,0,0,-1,1,1,1,-1,1];
    glm_fit_high.mu_n=mu0(:);
    glm_fit_low.mu_n=mu0(:);
    
    %clear MSE_high_stakes returns_high_stakes
    %clear MSE_low_stakes returns_low_stakes
    
    %for rep=1:nr_reps
    tic()
    [glm_fit_low(c),MSE_low_stakes(:,c),returns_low_stakes(:,c)]=...
        BayesianSARSAQ(low_stakes,feature_extractor,nr_training_episodes,glm_fit_low);
    disp(['Repetition ',int2str(c),' took ',int2str(round(toc()/60)),' minutes.'])
    
    
    tic()
    [glm_fit_high(c),MSE_high_stakes(:,c),returns_high_stakes(:,c)]=...
        BayesianSARSAQ(high_stakes,feature_extractor,nr_training_episodes,glm_fit_high);
    disp(['Repetition ',int2str(c),' took ',int2str(round(toc()/60)),' minutes.'])
    
    %end
    %{
    bin_width=200;
    clear avg_returns_high_stakes sem_avg_return_high_stakes
    clear avg_RMSE_high_stakes sem_RMSE_high_stakes
    
    nr_reps=1;
    for r=1:nr_reps
        [avg_returns_high_stakes(:,r),sem_avg_return_high_stakes(:,r)]=binnedAverage(returns_high_stakes(:,r),bin_width)
        [avg_RMSE_high_stakes(:,r),sem_RMSE_high_stakes(:,r)]=binnedAverage(sqrt(MSE_high_stakes(:,r)),bin_width);
        
        [avg_returns_low_stakes(:,r),sem_avg_return_low_stakes(:,r)]=binnedAverage(returns_low_stakes(:,r),bin_width)
        [avg_RMSE_low_stakes(:,r),sem_RMSE_low_stakes(:,r)]=binnedAverage(sqrt(MSE_low_stakes(:,r)),bin_width);
        
    end
    best_run_high_stakes=argmax(avg_returns_high_stakes(end,:));
    best_run_low_stakes=argmax(avg_returns_low_stakes(end,:));
    
    nr_episodes=size(returns_high_stakes,1);
    episode_nrs=(bin_width:bin_width:nr_episodes)';
    
    figure()
    subplot(2,1,1)
    errorbar(episode_nrs,avg_returns_high_stakes,sem_avg_return_high_stakes,'g-o','LineWidth',2), hold on
    set(gca,'FontSize',16)
    xlabel('Episode','FontSize',16)
    ylabel('R_{total}','FontSize',16),
    title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
    %ylim([0,10])
    xlim([0,nr_episodes+5])
    %hold on
    %plot(smooth(R_total,100),'r-')
    %legend('RMSE','R_{total}')
    xlabel('#Episodes','FontSize',16)
    subplot(2,1,2)
    errorbar(episode_nrs,avg_RMSE_high_stakes,sem_RMSE_high_stakes,'g-o','LineWidth',2), hold on
    xlim([0,nr_episodes+5])
    set(gca,'FontSize',16)
    xlabel('Episode','FontSize',16)
    ylabel('RMSE','FontSize',16),
    %legend('with PR','without PR')
    %hold on
    %plot(smooth(R_total,100),'r-')
    %legend('RMSE','R_{total}')
    xlabel('#Episodes','FontSize',16)
    
    figure()
    subplot(2,1,1)
    errorbar(episode_nrs,avg_returns_low_stakes,sem_avg_return_low_stakes,'g-o','LineWidth',2), hold on
    set(gca,'FontSize',16)
    xlabel('Episode','FontSize',16)
    ylabel('R_{total}','FontSize',16),
    title('Semi-gradient SARSA (Q) in Mouselab Task, Low Stakes','FontSize',18)
    %ylim([0,10])
    xlim([0,nr_episodes+5])
    %hold on
    %plot(smooth(R_total,100),'r-')
    %legend('RMSE','R_{total}')
    xlabel('#Episodes','FontSize',16)
    subplot(2,1,2)
    errorbar(episode_nrs,avg_RMSE_low_stakes,sem_RMSE_low_stakes,'g-o','LineWidth',2), hold on
    xlim([0,nr_episodes+5])
    set(gca,'FontSize',16)
    xlabel('Episode','FontSize',16)
    ylabel('RMSE','FontSize',16),
    %legend('with PR','without PR')
    %hold on
    %plot(smooth(R_total,100),'r-')
    %legend('RMSE','R_{total}')
    xlabel('#Episodes','FontSize',16)
    
        
    feature_names={'const','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)',...%'remaining time',...
        'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
        'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)',...%'last click decision',...
        'observation','decision' %'early decision',' %'observation'
        };
    
    weights_high_stakes=[glm_fit_low(:).mu_n];
    figure()
    bar(weights_high_stakes(:,best_run_high_stakes)),
    %bar(w)
    %ylim([0,0.3])
    set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
    set(gca,'XTickLabelRotation',45,'FontSize',16)
    ylabel('Learned Weights','FontSize',16)
    title(['Bayesian SARSA without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)
    
    %}
    
    [R_total_high_range,problems_high_range,states_high_range,actions_high_range,indices_high_range(c)]=...
        inspectPolicy(high_stakes,feature_extractor,glm_fit_high(c),0,100)
    
    [R_total_low_range,problems_low_range,states_low_range,actions_low_range,indices_low_range(c)]=...
        inspectPolicy(high_stakes,feature_extractor,glm_fit_low(c),0,100)
    
    avg_nr_acquisitions_low(c)=mean(indices_low_range(c).nr_acquisitions);
    avg_nr_acquisitions_high(c)=mean(indices_high_range(c).nr_acquisitions);
    
    glm_fitted_low(c)=glm_fit_low(c)
    glm_fitted_high(c)=glm_fit_high(c)
end

glm_fitted_low
glm_fitted_high
%%
clear
load('optimal_policy_high_stakes_27-Jan-2017.mat')
load('optimal_policy_low_stakes_27-Jan-2017.mat')


[R_total_low_range,problems_low_range,states_low_range,actions_low_range,indices_low_range]=...
    inspectPolicy(low_stakes,feature_extractor,glm_low_stakes(best_run_low_stakes),0,1000)

[R_total_high_range,problems_high_range,states_high_range,actions_high_range,indices_high_range]=...
    inspectPolicy(high_stakes,feature_extractor,glm_high_stakes(best_run_high_stakes),0,1000)

%The model predicted a significant effect of dispersion on the
%prioritization of the most probable outcome on high stakes
%problems (88% vs. 45%; t(998)=40.86 p<0.0001) but not on low stakes problems (47% vs. 46%; t(175)=0.19, p=0.85).

figure()
bar([indices_low_range.effect_of_dispersion.means,indices_high_range.effect_of_dispersion.means]')
set(gca,'XTickLabel',{'Low Stakes','High Stakes'},'FontSize',16)
ylabel('% Acquisitions on Most Probable Outcome','FontSize',16)
legend('High Dispersion','Low Dispersion')

stats.acquisitions=[[mean(indices_low_range.nr_acquisitions),mean(indices_high_range.nr_acquisitions)];...
    [sem(indices_low_range.nr_acquisitions'),sem(indices_high_range.nr_acquisitions')]]/28*100

stats.PTPROB=[[nanmean(indices_low_range.PTPROB),nanmean(indices_high_range.PTPROB)];...
    [sem(indices_low_range.PTPROB'),sem(indices_high_range.PTPROB')]]*100

stats.var_attribute=[[mean(indices_low_range.var_attribute),mean(indices_high_range.var_attribute)];...
    [sem(indices_low_range.var_attribute'),sem(indices_high_range.var_attribute')]]

stats.var_alternative=[[mean(indices_low_range.var_alternative),mean(indices_high_range.var_alternative)];...
    [sem(indices_low_range.var_alternative'),sem(indices_high_range.var_alternative')]]

stats.pattern=[[nanmean(indices_low_range.pattern),nanmean(indices_high_range.pattern)];...
    [sem(indices_low_range.pattern'),sem(indices_high_range.pattern')]]*100

stats.percent_optimal=[[mean(indices_low_range.percent_optimal_EV),mean(indices_high_range.percent_optimal_EV)];...
    [sem(indices_low_range.percent_optimal_EV'),sem(indices_high_range.percent_optimal_EV')]]

figure()
barwitherr([stats.acquisitions(2,:)',stats.percent_optimal(2,:)',stats.PTPROB(2,:)',stats.pattern(2,:)']',...
    [stats.acquisitions(1,:)',stats.percent_optimal(1,:)',stats.PTPROB(1,:)',stats.pattern(1,:)']')
set(gca,'FontSize',16)
legend('Low Stakes','High Stakes')
set(gca,'XTickLabel',{'% Acquisitions','% Optimal',...
    '% Prioritization of argmax p(o)','Alternative-based processing'},...
    'XTickLabelRotation',45)
title('Model Predictions','FontSize',16)
ylabel('Percent','FontSize',16)

%inspect the choices of the high-stakes policy
high_stakes.nr_gambles=7;
for e=1:100
    [[zeros(1,high_stakes.nr_gambles+1);squeeze(problems_high_range(:,:,e))],actions_high_range(:,:,e)]
    pause()
end

%inspect the choices of the low-stakes policy
for e=1:100
    [[zeros(1,low_stakes.nr_gambles+1);squeeze(problems_low_range(:,:,e))],actions_low_range(:,:,e)]
    pause()
end

%% Evaluate a policy based on the myopic VOC

experiment=MouselabExperimentSimplified();
experiment.action_features=[1];

cost_values=[0.05:0.05:0.2,0.5]/3600;

for c=5%1:numel(cost_values)
    cost_per_click=cost_values(c);
    
    high_stakes=experiment.setPayoffRange(0.01,9.99);
    high_stakes.payoff_ranges=[0.01,9.99];
    high_stakes.time_per_click=1;
    high_stakes.time_cost_per_sec=cost_per_click;
    
    low_stakes=experiment.setPayoffRange(0.00,0.25);
    low_stakes.payoff_ranges=[0.00,0.25];
    low_stakes.time_per_click=1;
    low_stakes.time_cost_per_sec=cost_per_click;
    
    
    sigma0=10^-12;
    glm=BayesianGLM(1,sigma0);
    glm.mu_0=1;
    glm.mu_n=1;
    %feature_extractor=@(s,a,mdp) [mdp.myopicVOC(state,action)]
    feature_extractor=@(s,a,mdp) [mdp.regretReductionMinusCost(state,action)]
    %feature_extractor=@(s,a,mdp) [mdp.twoStepVOC(state,action)];
    
    
    tic()
    [R_total_low_range,problems_low_range{c},states_low_range{c},actions_low_range{c},indices_low_range(c)]=...
        inspectPolicy(low_stakes,feature_extractor,glm,0,200)
    toc()
    
    tic()
    [R_total_high_range,problems_high_range{c},states_high_range{c},actions_high_range{c},indices_high_range(c)]=...
        inspectPolicy(high_stakes,feature_extractor,glm,0,200)
    toc()
    
    avg_nr_acquisitions_high(c)=nanmean(indices_high_range(c).nr_acquisitions);
    avg_nr_acquisitions_low(c)=nanmean(indices_low_range(c).nr_acquisitions);
    sem_nr_acquisitions_high(c)=sem(indices_high_range(c).nr_acquisitions(:));
    sem_nr_acquisitions_low(c)=sem(indices_low_range(c).nr_acquisitions(:));
    
end

nr_acquisitions_people_low=11.99;
nr_acquisitions_people_high=15.42;
avg_nr_acquisitions_people=mean([nr_acquisitions_people_low;nr_acquisitions_people_high])

SE=(mean([avg_nr_acquisitions_low;avg_nr_acquisitions_high])-avg_nr_acquisitions_people).^2;
estimate=argmin(SE)

indices_high_range(estimate).pattern=-indices_high_range(estimate).pattern;
indices_low_range(estimate).pattern=-indices_low_range(estimate).pattern;

DVs={'nr_acquisitions','pattern','PTPROB','percent_optimal_EV'};
stakes_values=0:1;
compensatoriness_values=1:-1:0;
DV_labels={'Nr. Acquisitions','Outcome-Based Processing','Acq. on Most Probable Outcome','Relative Performance'}

fig=figure()
for dv=1:numel(DVs)
    data_low=indices_low_range(estimate).(DVs{dv});
    data_high=indices_high_range(estimate).(DVs{dv});
    
    DV_data=[data_low,data_high];
    
    for dispersion=1:2
        comp_value=compensatoriness_values(dispersion);
        
        results.(DVs{dv}).means(1:2,dispersion)=...
            [nanmean(data_low(indices_low_range(estimate).was_compensatory(:)==comp_value));...
             nanmean(data_high(indices_high_range(estimate).was_compensatory==comp_value))];
        
        results.(DVs{dv}).sems(1:2,dispersion)=...
            [sem(data_low(indices_low_range(estimate).was_compensatory(:)==comp_value)');...
             sem(data_high(indices_high_range(estimate).was_compensatory(:)==comp_value)')];

    end
    
    subplot(2,2,dv)
    barwitherr(results.(DVs{dv}).sems,results.(DVs{dv}).means)
    title(DV_labels{dv},'FontSize',18)
    set(gca,'XTickLabel',{'Low Stakes','High Stakes'},'FontSize',16)
    legend('Low Dispersion','High Dispersion')
    
    has_high_dispersion=[indices_low_range(estimate).was_compensatory,...
        indices_high_range(estimate).was_compensatory];
    has_high_stakes=[zeros(size(indices_low_range(estimate).was_compensatory)),...
        zeros(size(indices_high_range(estimate).was_compensatory))];
    
    include=not(isnan(DV_data(:)));
    subject_nr=repmat((1:200)',[1,2,10]);
    [p_values(:,dv),anova_table(:,:,dv),anova_stats(dv)]=anovan(DV_data(include(:)),...
        {has_high_dispersion(include(:)),has_high_stakes(include(:))},...
        'varnames',{'Dispersion','Stakes'})

end
tightfig

%inspect the choices of the high-stakes policy
high_stakes.nr_gambles=7;
for e=1:100
    [[zeros(1,high_stakes.nr_gambles+1);squeeze(problems_high_range{estimate}(:,:,e))];...
       [[0;problems_high_range{estimate}(:,1,e)], actions_high_range{estimate}(:,:,e)]]
    pause()
end

%inspect the choices of the low-stakes policy
for e=1:100
    [[zeros(1,low_stakes.nr_gambles+1);squeeze(problems_low_range(:,:,e))];actions_low_range(:,:,e)]
    pause()
end

%% Learn policies with a regret reduction feature
clear

experiment=MouselabExperiment2();

high_stakes=experiment.setPayoffRange(0.01,9.99);
high_stakes.payoff_ranges=[0.01,9.99];
high_stakes.time_per_click=1;
high_stakes.time_cost_per_sec=0.25/3600;

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);


sigma0=0.1;
glm_fit_high=BayesianGLM(14,sigma0);
mu0=[0,0.5,0.5,-0.5,-0.5,0,0,1,1,1,1,1,0,1];
glm_fit_high.mu_n=mu0(:);

time_costs=[0.0067,0.0233,0.0816,0.2856];

nr_training_episodes=2000;
nr_reps=4;
parfor rep=1:nr_reps
    high_stakes_temp=high_stakes;
    high_stakes_temp.time_cost_per_sec=time_costs(rep);
    tic()
    [glm_high_stakes(rep),MSE_high_stakes(:,rep),returns_high_stakes(:,rep)]=...
        BayesianSARSAQ(high_stakes_temp,feature_extractor,nr_training_episodes,glm_fit_high);
    disp(['Repetition ',int2str(rep),' took ',int2str(round(toc()/60)),' minutes.'])
end

bin_width=200;
for r=1:nr_reps
    [avg_returns_high_stakes(:,r),sem_avg_return_high_stakes(:,r)]=binnedAverage(returns_high_stakes(:,r),bin_width)
    [avg_RMSE_high_stakes(:,r),sem_RMSE_high_stakes(:,r)]=binnedAverage(sqrt(MSE_high_stakes(:,r)),bin_width);
end
best_run_high_stakes=argmax(avg_returns_high_stakes(end,:));


avg_MSE_high_stakes=mean(MSE_high_stakes(:,3),2);

R_total_high_stakes=mean(returns_high_stakes(:,3),2);

nr_episodes=size(R_total_high_stakes,1);
bin_width=100;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE_high_stakes,sem_RMSE_high_stakes]=binnedAverage(sqrt(avg_MSE_high_stakes),bin_width);
[avg_R_high_stakes,sem_R_high_stakes]=binnedAverage(R_total_high_stakes,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R_high_stakes,sem_R_high_stakes,'g-o','LineWidth',2), hold on
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
%ylim([0,10])
xlim([0,nr_episodes+5])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE_high_stakes,sem_RMSE_high_stakes,'g-o','LineWidth',2), hold on
xlim([0,nr_episodes+5])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
%legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'const','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)',...%'remaining time',...
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)',...%'last click decision',...
    'observation','regret reduction' %'early decision',' %'observation', '#observations'
    };

weights_high_stakes=[glm_high_stakes(1:nr_reps).mu_n];
figure()
bar(weights_high_stakes),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Bayesian SARSA without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)

for r=1:nr_reps
    high_stakes.time_cost_per_sec=time_costs(r)
    [R_total_high_range{r},problems_high_range,states_high_range{r},actions_high_range{r},indices_high_range(r)]=...
        inspectPolicy(high_stakes,feature_extractor,glm_high_stakes(r),0,100)
    avg_nr_acquisitions(r)=mean(indices_high_range(r).nr_acquisitions)
end


save(['optimal_policy_high_stakes_',date()])

%% fit time cost to low stakes problems/ simulate performance on low stakes problems with estimate from high-stakes problems
clear

experiment=MouselabExperiment2();

low_stakes=experiment.setPayoffRange(0.00,0.25);
low_stakes.payoff_ranges=[0.00,0.25];
low_stakes.time_per_click=1;

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);


sigma0=0.1;
glm_fit_low=BayesianGLM(14,sigma0);
mu0=[0,0.5,0.5,-0.5,-0.5,0,0,1,1,1,1,1,0,1];
glm_fit_low.mu_n=mu0(:);

time_costs=[0.0233, 0.0354, 0.0537,0.0816];

nr_training_episodes=2000;
nr_reps=4;
parfor rep=1:nr_reps
    low_stakes_temp=low_stakes;
    low_stakes_temp.time_cost_per_sec=time_costs(rep);
    tic()
    [glm_low_stakes(rep),MSE_low_stakes(:,rep),returns_low_stakes(:,rep)]=...
        BayesianSARSAQ(low_stakes_temp,feature_extractor,nr_training_episodes,glm_fit_low);
    disp(['Repetition ',int2str(rep),' took ',int2str(round(toc()/60)),' minutes.'])
end

bin_width=200;
for r=1:nr_reps
    [avg_returns_low_stakes(:,r),sem_avg_return_low_stakes(:,r)]=binnedAverage(returns_low_stakes(:,r),bin_width)
    [avg_RMSE_low_stakes(:,r),sem_RMSE_low_stakes(:,r)]=binnedAverage(sqrt(MSE_low_stakes(:,r)),bin_width);
end
best_run_low_stakes=argmax(avg_returns_low_stakes(end,:));


avg_MSE_low_stakes=mean(MSE_low_stakes(:,3),2);

R_total_low_stakes=mean(returns_low_stakes(:,3),2);

nr_episodes=size(R_total_low_stakes,1);
bin_width=100;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE_low_stakes,sem_RMSE_low_stakes]=binnedAverage(sqrt(avg_MSE_low_stakes),bin_width);
[avg_R_low_stakes,sem_R_low_stakes]=binnedAverage(R_total_low_stakes,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R_low_stakes,sem_R_low_stakes,'g-o','LineWidth',2), hold on
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
%ylim([0,10])
xlim([0,nr_episodes+5])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE_low_stakes,sem_RMSE_low_stakes,'g-o','LineWidth',2), hold on
xlim([0,nr_episodes+5])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
%legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'const','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)',...%'remaining time',...
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)',...%'last click decision',...
    'observation','regret reduction' %'early decision',' %'observation', '#observations'
    };

weights_low_stakes=[glm_low_stakes(1:nr_reps).mu_n];
figure()
bar(weights_low_stakes),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Bayesian SARSA without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)

for r=1:nr_reps
    low_stakes.time_cost_per_sec=time_costs(r)
    [R_total_low_range{r},problems_low_range,states_low_range{r},actions_low_range{r},indices_low_range(r)]=...
        inspectPolicy(low_stakes,feature_extractor,glm_low_stakes(r),0,100)
    avg_nr_acquisitions_low(r)=mean(indices_low_range(r).nr_acquisitions)
end


save(['optimal_policy_low_stakes_',date()])

%% refine the policies by a few additional training episodes
clear
load(['optimal_policy_high_stakes_29-Jan-2017'])
load(['optimal_policy_low_stakes_29-Jan-2017'])

selected_policies=[glm_low_stakes(2);glm_high_stakes(2)];
low_stakes.time_cost_per_sec=0.03;
high_stakes.time_cost_per_sec=0.03;
problems=[low_stakes;high_stakes]

nr_training_episodes=2000;
nr_reps=numel(selected_policies);
parfor rep=1:nr_reps
    tic()
    [selected_policies(rep),MSE_low_stakes(:,rep),returns_low_stakes(:,rep)]=...
        BayesianSARSAQ(problems(rep),feature_extractor,nr_training_episodes,selected_policies(rep));
    disp(['Repetition ',int2str(rep),' took ',int2str(round(toc()/60)),' minutes.'])
end

save(['optimal_polies_',date()])

feature_names={'const','E[max mu]','mu(a*)','sigma(a*)','sigma(max EV)',...%'remaining time',...
    'mu(b)','sigma(b)','Expected Regret', ... %'EV',...
    'myopic VOC', 'sigma(g)', 'P(o)','max mu - mu(g)',...%'last click decision',...
    'observation','regret reduction' %'early decision',' %'observation', '#observations'
    };

weights=[selected_policies(1:2).mu_n];
figure()
bar(weights),
%bar(w)
%ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title(['Bayesian SARSA without PR, ',int2str(nr_episodes),' episodes'],'FontSize',18)


%% simulate experiment with the two policies and compare
clear

cost_per_click=0.03;
load(['optimal_policies_30-Jan-2017.mat'])


%load(['optimal_policy_high_stakes_29-Jan-2017'])
%load(['optimal_policy_low_stakes_29-Jan-2017'])

%estimate_high=argmin(abs(time_costs_high_stakes-cost_per_click));
%estimate_low=argmin(abs(time_costs_low_stakes-cost_per_click));

%experiment=MouselabExperiment2();

%low_stakes=experiment.setPayoffRange(0.00,0.25);
%low_stakes.payoff_ranges=[0.00,0.25];
%low_stakes.time_per_click=1;
%low_stakes.time_cost_per_sec=cost_per_click;

%high_stakes=experiment.setPayoffRange(0.01,9.99);
%high_stakes.payoff_ranges=[0.01,9.99];
%high_stakes.time_per_click=1;
%high_stakes.time_cost_per_sec=cost_per_click;

%feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

[R_total_low_stakes,problems_low_stakes,states_low_stakes,actions_low_stakes,indices_low_stakes]=...
    inspectPolicy(problems(1),feature_extractor,selected_policies(1),0,200)
[R_total_high_stakes,problems_high_stakes,states_high_stakes,actions_high_stakes,indices_high_stakes]=...
    inspectPolicy(problems(2),feature_extractor,selected_policies(2),0,200)

indices_low_stakes.pattern=-indices_low_stakes.pattern;
indices_high_stakes.pattern=-indices_high_stakes.pattern;
DVs={'nr_acquisitions','pattern','PTPROB','percent_optimal_EV'};
stakes_values=0:1;
compensatoriness_values=1:-1:0;
DV_labels={'Nr. Acquisitions','Outcome-Based Processing','Acq. on Most Probable Outcome','Relative Performance'}

fig=figure()
for dv=1:numel(DVs)
    data_low=indices_low_stakes.(DVs{dv});
    data_high=indices_high_stakes.(DVs{dv});
    
    DV_data=[data_low,data_high];
    
    for dispersion=1:2
        comp_value=compensatoriness_values(dispersion);
        
        results.(DVs{dv}).means(1:2,dispersion)=...
            [nanmean(data_low(indices_low_stakes.was_compensatory(:)==comp_value));...
             nanmean(data_high(indices_high_stakes.was_compensatory==comp_value))];
        
        results.(DVs{dv}).sems(1:2,dispersion)=...
            [sem(data_low(indices_low_stakes.was_compensatory(:)==comp_value)');...
             sem(data_high(indices_high_stakes.was_compensatory(:)==comp_value)')];

    end
    
    s_id(dv)=subplot(2,2,dv)
    barwitherr(results.(DVs{dv}).sems,results.(DVs{dv}).means)
    title(DV_labels{dv},'FontSize',18)
    set(gca,'XTickLabel',{'Low Stakes','High Stakes'},'FontSize',16)
    legend('Low Dispersion','High Dispersion')
    
    has_high_dispersion=[indices_low_stakes.was_compensatory,...
        indices_high_stakes.was_compensatory];
    has_high_stakes=[zeros(size(indices_low_stakes.was_compensatory)),...
        ones(size(indices_high_stakes.was_compensatory))];
    
    include=not(isnan(DV_data(:)));
    subject_nr=repmat((1:200)',[1,2,10]);
    [p_values(:,dv),anova_table(:,:,dv),anova_stats(dv)]=anovan(DV_data(include(:)),...
        {has_high_dispersion(include(:)),has_high_stakes(include(:))},...
        'varnames',{'Dispersion','Stakes'},'model',3)
    
    disp(DV_labels{dv}),pause()
end
tightfig

%inspect the choices of the high-stakes policy
high_stakes.nr_gambles=7;

%inspect the choices of the low-stakes policy
for e=1:100
    [[zeros(1,low_stakes.nr_gambles+1);squeeze(problems_low_stakes(:,:,e))];...
        [[0;problems_low_stakes(:,1,e)], actions_low_stakes(:,:,e)]]
    
    max_prob(e)=max(problems_low_stakes(:,1,e));
    compensatory(e)=max_prob(e)<0.75;
    most_probable=argmax(problems_low_stakes(:,1,e));
    less_probable=setdiff(1:problems(1).nr_outcomes,most_probable);
    others=actions_low_stakes(1+less_probable,2:end,e);
    consistent_with_TTB(e)=and(all(not(isnan(actions_low_stakes(1+most_probable,:,e)))),...
        all(isnan(others(:))))
    consistent_with_SAT_TTB(e)=and(and(not(all(isnan(actions_low_stakes(1+most_probable,:,e)))),...
        all(isnan(others(:)))),any(isnan(actions_low_stakes(1+most_probable,:,e))))
        
    inspected_payoff_indices=find(not(isnan(actions_low_stakes(2:end,:,e))));
    payoffs=problems_low_stakes(:,2:end,e);
    
    if numel(inspected_payoff_indices)>0
        inspected_payoffs=payoffs(inspected_payoff_indices);
        max_inspected_payoff(e)=max(inspected_payoffs);
        max_probability(e)=max(problems_low_stakes(:,1,e));
        max_EV(e)=max(inspected_payoffs)*max_probability(e);
        random_choice(e)=false;
    else
        random_choice(e)=true;
        inspected_payoffs=[];
    end
    
    if numel(inspected_payoffs)>1
        max_inspected_payoff_before_last_move(e)=max(inspected_payoffs(1:end-1))
        max_EV_before_last_move(e)=max_probability(e)*max_inspected_payoff_before_last_move(e);
    end

    
end

%The optimal policy chose randomly
mean(random_choice(compensatory))
mean(random_choice(~compensatory))
%On the low stakes problems, the model produced the random choice strategy
%56% of the time when the the dispersion of outcome probabilities was low
%but never used it when the dispersion of the outcome probabilities was
%high.

%The optimal policy chose randomly
mean(consistent_with_TTB(compensatory))
mean(consistent_with_TTB(~compensatory))
%The model produced TTB on 100% of the non-compensatory low-stakes
%problems, but on only 20% of the compensatory low stakes problems.

mean(consistent_with_SAT_TTB(compensatory))
mean(consistent_with_SAT_TTB(~compensatory))
%The model produced SAT-TTB on 24% of the compensatory low-stakes problems
%but not on the non-compensatory low-stakes problems.

mean(max_inspected_payoff_before_last_move(consistent_with_TTB))
mean(max_EV_before_last_move(consistent_with_TTB))
mean(max_probability(consistent_with_TTB))

min(max_EV(consistent_with_TTB))
min(max_inspected_payoff(consistent_with_TTB))

mean(max_EV(consistent_with_SAT_TTB))
mean(max_inspected_payoff(consistent_with_SAT_TTB)) %guaranteed to be better than average


%% 

load optimal_policies_30-Jan-2017
%problems_low_stakes=problems(1)
%problems_high_stakes=problems(2)

for e=1:100
    [[zeros(1,high_stakes.nr_gambles+1);squeeze(problems_high_stakes(:,:,e))];...
       [[0;problems_high_stakes(:,1,e)], actions_high_stakes(:,:,e)]]
    pause()
    max_prob(e)=max(problems_high_stakes(:,1,e));
    compensatory(e)=max_prob(e)<0.75;
    most_probable=argmax(problems_high_stakes(:,1,e));
    less_probable=setdiff(1:4,most_probable);
    others=actions_high_stakes(1+less_probable,2:end,e);
    consistent_with_TTB(e)=and(all(not(isnan(actions_high_stakes(1+most_probable,:,e)))),...
        all(isnan(others(:))))
    consistent_with_SAT_TTB(e)=and(and(not(all(isnan(actions_high_stakes(1+most_probable,:,e)))),...
        all(isnan(others(:)))),any(isnan(actions_high_stakes(1+most_probable,:,e))))
    
    acquisitions=actions_high_stakes(2:end,:,e)
    consistent_with_WADD(e)=all(not(isnan(acquisitions(:))))
    
    inspected_payoff_indices=find(not(isnan(actions_high_stakes(2:end,:,e))));
    payoffs=problems_high_stakes(:,2:end,e);
    
    if numel(inspected_payoff_indices)>0
        inspected_payoffs=payoffs(inspected_payoff_indices);
        max_inspected_payoff(e)=max(inspected_payoffs);
        max_probability(e)=max(problems_high_stakes(:,1,e));
        max_EV(e)=max(inspected_payoffs)*max_probability(e);
        random_choice(e)=false;
    else
        random_choice(e)=true;
        inspected_payoffs=[];
    end
    
    if numel(inspected_payoffs)>1
        max_inspected_payoff_before_last_move(e)=max(inspected_payoffs(1:end-1))
        max_EV_before_last_move(e)=max_probability(e)*max_inspected_payoff_before_last_move(e);
    end
       
end

%The optimal policy chose randomly
mean(random_choice(compensatory))
mean(random_choice(~compensatory))
%On the low stakes problems, the model produced the random choice strategy
%56% of the time when the the dispersion of outcome probabilities was low
%but never used it when the dispersion of the outcome probabilities was
%high.

%The optimal policy chose randomly
mean(consistent_with_TTB(compensatory))
mean(consistent_with_TTB(~compensatory))
%The model produced TTB on 100% of the non-compensatory low-stakes
%problems, but on only 20% of the compensatory low stakes problems.

mean(consistent_with_SAT_TTB(compensatory))
mean(consistent_with_SAT_TTB(~compensatory))
%The model produced SAT-TTB on 24% of the compensatory low-stakes problems
%but not on the non-compensatory low-stakes problems.

