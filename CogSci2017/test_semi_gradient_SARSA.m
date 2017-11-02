%test semi-gradient RL algorithm that learns an approximation to V

addpath('~/Dropbox/PhD/MatlabTools/')

nr_actions=2;
nr_states=2;
gamma=1;
epsilon=0.1;

mdp=testMDP(nr_actions,gamma);
nr_episodes=1000;
feature_extractor=@(s) [ones(1,size(s,2)); s+1];

[w,avg_MSE]=semiGradientSARSA(mdp,feature_extractor,nr_episodes,epsilon);

figure(),
plot(avg_MSE)
xlabel('Episode','FontSize',16)
ylabel('Average MSE','FontSize',16)

%% test semi-gradient SARSA that learns Q
addpath('~/Dropbox/PhD/MatlabTools/')

nr_actions=2;
nr_states=2;
gamma=1;

mdp=testMDP(nr_actions,gamma);

epsilon=0.1;
nr_episodes=10000;
feature_extractor=@(s,a) [ones(1,size(s,2)); s+1; a];

[w,avg_MSE]=semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,epsilon);

figure(),
plot(avg_MSE)
xlabel('Episode','FontSize',16)
ylabel('Average MSE','FontSize',16)

%% test semi-gradient SARSAQ on Mouselab MDP
total_nr_clicks=10;
non_compensatoriness=0.5;
mdp=MouselabMDP(non_compensatoriness,total_nr_clicks);

epsilon=0.05;
nr_episodes=100;
feature_extractor=@(s,a) mdp.extractStateActionFeatures(s,a);

[w,avg_MSE,R_total]=semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,epsilon);

%% test semi-gradient V-learning
clear
alpha_clicks_per_cell=100;
beta_clicks_per_cell=100;
non_compensatoriness=1;
mdp=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell);

epsilon=0.05;
nr_episodes=750;
feature_extractor=@(s,a) mdp.extractStateFeatures(s);

[w,avg_MSE,R_total]=LinearTD0(mdp,feature_extractor,nr_episodes);

figure()
subplot(2,1,1)
plot(smooth(sqrt(avg_MSE),100,'loess')),ylabel('RMSE','FontSize',16)
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
bar(w),set(gca,'XTickLabel',{'mu(a*)','sigma(a*)','sigma(max EV)','remaining #clicks','mu(b)',...
    'sigma(b)','Expected Regret','E[max mu]'})
set(gca,'XTickLabelRotation',45)
ylabel('Learned Weights','FontSize',16)


[w,avg_MSE,R_total]=semiGradientSARSA(mdp,feature_extractor,nr_episodes,epsilon);

figure()
subplot(3,1,1)
plot(smooth(R_total,20)),ylabel('R_{total}','FontSize',16)
xlabel('#Episodes','FontSize',16)
subplot(3,1,2)
plot(smooth(sqrt(avg_MSE),25)),ylabel('RMSE','FontSize',16)
xlabel('#Episodes','FontSize',16)
subplot(3,1,3)
bar(w),set(gca,'XTickLabel',{'mu(a*)','sigma(a*)','sigma(max EV)','remaining #clicks','mu(b)',...
    'sigma(b)','Expected Regret','E[max mu]'})
set(gca,'XTickLabelRotation',45)
ylabel('Learned Weights','FontSize',16)