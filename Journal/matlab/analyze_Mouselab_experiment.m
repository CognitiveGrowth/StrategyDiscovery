%analyze Mouselab experiment
addpath('~/Dropbox/PhD/MatlabTools/')
addpath('~/Dropbox/PhD/MatlabTools/parse_json/')
clear

load(['../data/fullyRevealed_100uniqueTrials/Mouselab_data_Experiment.mat'])
% load(['../data/3conditions_300subjects/Mouselab_data_Experiment_condition2.mat'])
% load(['../data/03242018/Mouselab_data_Experiment2.mat']) 3conditions_300subjects

% experiment_nr = 2;
% try
%     load(['../data/Mouselab_data_Experiment',int2str(experiment_nr),'.mat'])
% catch
%     import_Mouselab_data
%     pay_bonus
% end

for s=1:numel(data_by_sub)
    if data.basic_info{s}.isFullyRevealed == 0 && data.basic_info{s}.isHiddenProbability == 0
        data.condition(s) = 1;
    elseif data.basic_info{s}.isFullyRevealed == 1 && data.basic_info{s}.isHiddenProbability == 0
        data.condition(s) = 2;
    elseif data.basic_info{s}.isFullyRevealed == 0 && data.basic_info{s}.isHiddenProbability == 1
        data.condition(s) = 3;
    else
        error('!')
    end
    data.bonus_(s) = str2num(data.bonus{s});
end


avg_completion_time = mean([data.experimentTime{:}])/60
for s=1:numel(data_by_sub)
     bonus(s) = str2num(data_by_sub{s}.bonus);
end
avg_bonus = mean(bonus)

nr_blocks=numel(data.decision_problems{1});
nr_trials_per_block=numel(data.decision_problems{1}{1});

nr_outcomes=numel(data.decision_problems{1}{1}{1}.payoffs);
nr_gambles=numel(data.decision_problems{1}{1}{1}.payoffs{1});

for sub=1:numel(data_by_sub)
    
    if data_by_sub{sub}.decision_problems{1}{1}.mu{1}==5
        data.high_stakes(sub,1)=true;
        data.high_stakes(sub,2)=false;
    elseif data_by_sub{sub}.decision_problems{1}{1}.mu{1}==0.13
        data.high_stakes(sub,1)=false;
        data.high_stakes(sub,2)=true;
    end
    
    
    for b=1:nr_blocks
        
        for t=1:nr_trials_per_block
            temp=[data.decision_problems{sub}{b}{t}.revealed{:}];
            revealed=[temp{:}];
            
            data.chosen_gamble(sub,b,t)=str2num(data.decisions{sub}{b}{t});
            
            
            
            if data.high_stakes(sub,b)
                data.nr_acquisitions_high_stakes(sub,t)=sum(revealed);
            else
                data.nr_acquisitions_low_stakes(sub,t)=sum(revealed);
            end
            
            
            %record whether the probabilitie have high vs. low dispersion
            probabilities=[data_by_sub{sub}.decision_problems{b}{t}.probabilities{:}];
            if max(probabilities)>=0.8
                data.high_dispersion(sub,b,t)=true;
            else
                data.high_dispersion(sub,b,t)=false;
            end
            
            
            [~,p_sorted]=sort(probabilities,'descend');
            most_probable=p_sorted(1);
            second_most_probable=p_sorted(2);
            third_most_probable=p_sorted(3);
            data.percent_most_probable(sub,b,t)=sum([data.decision_problems{sub}{b}{t}.revealed{most_probable}{:}])/sum(revealed);
            
            for o=1:nr_outcomes
                data.acquisition_order(sub,b,t,o,:)=[data.decision_problems{sub}{b}{t}.reveal_order{o}{:}];
                data.probabilities(sub,b,t,:)=[data.decision_problems{sub}{b}{t}.probabilities{:}];
                for g=1:nr_gambles
                    data.payoff_matrices(sub,b,t,o,g)=str2num(data.decision_problems{sub}{b}{t}.payoffs{o}{g});
                end
            end
            data.EVs(sub,b,t,:)=squeeze(data.payoff_matrices(sub,b,t,:,:))'*probabilities(:);
            
            data.EV_chosen_gamble(sub,b,t)=data.EVs(sub,b,t,data.chosen_gamble(sub,b,t));
            
            data.relative_performance(sub,b,t)=data.EV_chosen_gamble(sub,b,t)/max(data.EVs(sub,b,t,:));
            
            nr_acquisitions=max(squeeze(data.acquisition_order(sub,b,t,:)));
            data.nr_acquisitions(sub,b,t)=nr_acquisitions;
            
            nr_moves_within_outcomes=0;
            nr_moves_within_alternatives=0;
            gamble=NaN(nr_acquisitions,1);
            outcome=NaN(nr_acquisitions,1);
            inspected_most_probable_outcome=NaN(nr_acquisitions,1);
            inspected_second_most_probable_outcome=NaN(nr_acquisitions,1);
            inspected_third_most_probable_outcome=NaN(nr_acquisitions,1);
            values_of_most_probable_outcome = squeeze(data.payoff_matrices(sub,b,t,most_probable,:));
            gamble_of_most_valueable_of_most_probable_outcome = argmax(values_of_most_probable_outcome);
            inspected_gamble_of_most_valueable_of_most_probable_outcome=NaN(nr_acquisitions,1);
            for a=1:nr_acquisitions
                [outcome(a),gamble(a)]=find(squeeze(data.acquisition_order(sub,b,t,:,:))==a);
                
                if outcome(a)==most_probable
                    inspected_most_probable_outcome(a)=true;
                else
                    inspected_most_probable_outcome(a)=false;
                end
                
                if outcome(a)==second_most_probable
                    inspected_second_most_probable_outcome(a)=true;
                else
                    inspected_second_most_probable_outcome(a)=false;
                end
                
                if outcome(a)==third_most_probable
                    inspected_third_most_probable_outcome(a)=true;
                else
                    inspected_third_most_probable_outcome(a)=false;
                end
                
                if gamble(a)==gamble_of_most_valueable_of_most_probable_outcome
                    inspected_gamble_of_most_valueable_of_most_probable_outcome(a)=true;
                else
                    inspected_gamble_of_most_valueable_of_most_probable_outcome(a)=false;
                end

                
                if a>1
                    if gamble(a)==gamble(a-1)
                        nr_moves_within_alternatives=nr_moves_within_alternatives+1;
                    elseif outcome(a)==outcome(a-1)
                        nr_moves_within_outcomes=nr_moves_within_outcomes+1;
                    end
                    
                end
            end            
            
            data.pattern(sub,b,t)=-(nr_moves_within_alternatives-nr_moves_within_outcomes)/(nr_moves_within_alternatives+nr_moves_within_outcomes);
            
            temp_acquisitions=squeeze(data.acquisition_order(sub,b,t,:,:));
            
            nr_acquisitions_by_alternative=sum(temp_acquisitions>0,1);
            nr_acquisitions_by_outcome=sum(temp_acquisitions>0,2);
            
            data.nr_gambles_inspected(sub,b,t)=sum(nr_acquisitions_by_alternative(:)>0);
            data.nr_outcomes_inspected(sub,b,t)=sum(nr_acquisitions_by_outcome(:)>0);
            
            data.var_alternatives(sub,b,t)=var(nr_acquisitions_by_alternative(:));
            data.var_outcomes(sub,b,t)=var(nr_acquisitions_by_outcome(:));
            data.inspected_most_probable{sub,b,t}=inspected_most_probable_outcome;
            data.consistent_with_SAT_TTB(sub,b,t)=nr_acquisitions>0 && and(all(inspected_most_probable_outcome),data.nr_gambles_inspected(sub,b,t)<nr_gambles);
            data.consistent_with_TTB(sub,b,t)=and(all(inspected_most_probable_outcome),data.nr_gambles_inspected(sub,b,t)==nr_gambles);
            data.consistent_with_SAT(sub,b,t)=and(and(data.nr_gambles_inspected(sub,b,t)<nr_gambles,...
                nr_moves_within_outcomes==data.nr_gambles_inspected(sub,b,t)-1),...
                nr_moves_within_alternatives==(data.nr_gambles_inspected(sub,b,t)*(nr_outcomes-1)));

            data.consistent_with_WADD(sub,b,t)=and(nr_acquisitions==nr_gambles*nr_outcomes,...
                nr_moves_within_outcomes==nr_gambles-1);
            
            data.consistent_with_SAT_TTB2(sub,b,t) = nr_gambles>sum(inspected_most_probable_outcome) && sum(inspected_most_probable_outcome)>0 && nr_gambles>sum(inspected_second_most_probable_outcome) && sum(inspected_second_most_probable_outcome)>0 && ((sum(inspected_most_probable_outcome)+sum(inspected_second_most_probable_outcome))==nr_acquisitions);
%             this TTB2plus is relaxed to include TTB2:
            data.consistent_with_TTB2plus(sub,b,t) = nr_acquisitions>=(2*nr_gambles) && sum(inspected_most_probable_outcome(1:2*nr_gambles))==nr_gambles && sum(inspected_second_most_probable_outcome(1:2*nr_gambles))==nr_gambles && nr_acquisitions<nr_gambles*nr_outcomes;
            data.consistent_with_SAT_TTB3(sub,b,t) = nr_gambles>sum(inspected_most_probable_outcome) && sum(inspected_most_probable_outcome)>0 && nr_gambles>sum(inspected_second_most_probable_outcome) && sum(inspected_second_most_probable_outcome)>0 && nr_gambles>sum(inspected_third_most_probable_outcome) && sum(inspected_third_most_probable_outcome)>0 && ((sum(inspected_most_probable_outcome)+sum(inspected_second_most_probable_outcome)+sum(inspected_third_most_probable_outcome))==nr_acquisitions);
            data.consistent_with_TTBplus(sub,b,t) = sum(inspected_most_probable_outcome(:))==nr_gambles...
                && sum(inspected_most_probable_outcome(1:nr_gambles))==nr_gambles ...
                && nr_acquisitions>nr_gambles  && nr_acquisitions<nr_gambles*nr_outcomes;
            data.consistent_with_random = nr_acquisitions==0;
            
%              [?TTB?, ?WADD?, ?SAT_TTB?, ?random_choice?, ?SAT_TTB2?, ?SAT?, ?TTB2_plus?, ?SAT_TTB3?, ?TTB_plus?, 'other']
% % old TTBplus definition
%             data.consistent_with_TTBplus(sub,b,t) = sum(inspected_most_probable_outcome(:))==nr_gambles...
%                 && sum(inspected_most_probable_outcome(1:nr_gambles))==nr_gambles ...
%                 && sum(inspected_gamble_of_most_valueable_of_most_probable_outcome(:))==nr_outcomes;
% % this is equivalent (sanity check):
%             data.consistent_with_TTBplus(sub,b,t) = nr_acquisitions>=(nr_gambles+nr_outcomes-1) &&...
%                 all(outcome(1:nr_gambles)==most_probable) &&...
%                 sum(gamble(nr_gambles+1:end)==gamble_of_most_valueable_of_most_probable_outcome)==(nr_outcomes-1);

            data.EV_chosen_gamble_equals_maxEV(sub,b,t) = (data.EV_chosen_gamble(sub,b,t) == max(squeeze(data.EVs(sub,b,t,:))));
            data.EVbest_minus_EVchosen_over_EVbest(sub,b,t) = (data.EV_chosen_gamble(sub,b,t)-max(squeeze(data.EVs(sub,b,t,:))))/max(squeeze(data.EVs(sub,b,t,:)));
            data.EV_chosen_rank(sub,b,t) = find(sort(squeeze(data.EVs(sub,b,t,:)),'descend')==data.EV_chosen_gamble(sub,b,t),1);
            
        end
        
        try
            outcomes=[data_by_sub{sub}.outcomes{:}];
            data.outcomes(sub,:)=outcomes;
            
        catch
            disp(['Missing data from subject ',int2str(sub)])
        end
    end
    data.avg_percent_most_probable(sub,1)=nanmean(data.percent_most_probable(sub,data.high_dispersion(sub,:)));
    data.avg_percent_most_probable(sub,2)=nanmean(data.percent_most_probable(sub,~data.high_dispersion(sub,:)));
    
    data.isFullyRevealed(sub,1) = data_by_sub{sub}.basic_info.isFullyRevealed;
end

data.consistent_with_random = data.nr_acquisitions == 0;

has_low_stakes=repmat(~data.high_stakes,[1,1,10]);
has_high_stakes=~has_low_stakes;
has_high_dispersion=data.high_dispersion;
has_low_dispersion=~has_high_dispersion;

%TODO: add estimated strategy frequencies
DVs={'nr_acquisitions','pattern','percent_most_probable','relative_performance'};
stakes_values=0:1;
dispersion_values=0:1;
DV_labels={'Nr. Acquisitions','Outcome-Based Processing','Acq. on Most Probable Outcome','Relative Performance'}

data.pattern=data.pattern*100;
data.percent_most_probable=data.percent_most_probable*100;
data.relative_performance=data.relative_performance*100;
ylabels={'Nr. Cells Inspected','%','%','% Optimal'}

fid=figure()
for dv=1:numel(DVs)
    dv_data=data.(DVs{dv});
    
    for stakes=1:2
        stakes_value=stakes_values(stakes);
        for dispersion=1:2
            dispersion_value=dispersion_values(dispersion);
            
            results.(DVs{dv}).means(stakes,dispersion)=...
                nanmean(dv_data(and(has_high_stakes(:)==stakes_value,...
                has_high_dispersion(:)==dispersion_value)));
            
            results.(DVs{dv}).sems(stakes,dispersion)=...
                sem(dv_data(and(has_high_stakes(:)==stakes_value,...
                has_high_dispersion(:)==dispersion_value)));
        end
    end
    
    subplot(2,2,dv)
    barwitherr(results.(DVs{dv}).sems,results.(DVs{dv}).means)
    title(DV_labels{dv},'FontSize',18)
    set(gca,'XTickLabel',{'Low Stakes','High Stakes'},'FontSize',16)
    legend('Low Dispersion','High Dispersion')
    ylabel(ylabels{dv},'FontSize',16)
    
    include=not(isnan(dv_data(:)));
    subject_nr=repmat((1:300)',[1,2,10]);
    [p_values(:,dv),anova_table(:,:,dv),anova_stats(dv)]=anovan(dv_data(include(:)),...
        {has_high_dispersion(include(:)),has_high_stakes(include(:)),subject_nr(include(:))},...
        'varnames',{'Dispersion','Stakes','Subject'},...
        'random',[3])
    
    disp(DV_labels{dv})
    %pause()
end
tightfig

[[mean(mean(data.nr_acquisitions_high_stakes,2)),mean(mean(data.nr_acquisitions_low_stakes,2))];...
    [sem(data.nr_acquisitions_high_stakes(:)),sem(data.nr_acquisitions_low_stakes(:))]]'
[h,p,ci,stats]=ttest(mean(data.nr_acquisitions_high_stakes,2)-mean(data.nr_acquisitions_low_stakes,2))
%Participants inspected significantly more outcomes on high stakes problems
%than on low stakes problems (t(199)=7.50, p<10^{-4}). However, the
%difference of $15.42 \pm 0.21$ vs. $11.99 \pm 0.20$ inspections was smaller than predicted by the
%optimal policy.
prioritization_high_dispersion=nanmean(data.percent_most_probable(data.high_dispersion(:)))
prioritization_low_dispersion=nanmean(data.percent_most_probable(~data.high_dispersion(:)))
%Participants prioritized the most probable outcome significantly more when
%the the dispersion of the outcome probabilities was high (70.55% of acquisitions) than when it was
%low (40.44% of acquisitions). This difference was statistically
%significant (t(196)=18.45, p<0.0001).
[h,p,ci,stats]=ttest(data.avg_percent_most_probable(:,1)-data.avg_percent_most_probable(:,2));


ld_hs = and(has_high_stakes(:), ~has_high_dispersion(:));
ld_ls = and(has_low_stakes(:), ~has_high_dispersion(:));
hd_hs = and(has_high_stakes(:), has_high_dispersion(:));
hd_ls = and(has_low_stakes(:), has_high_dispersion(:));

conditions = [hd_hs,hd_ls,ld_hs,ld_ls];
nr_trials_exp1 = sum([data.condition]==1)*5;
nr_trials_exp2 = sum([data.condition]==2)*5;
nr_trials_per_condition_exp1 = size(conditions,1)/4;

experiment1=repmat([data.condition]==1,[1,2,10]);
experiment2=repmat([data.condition]==2,[1,2,10]);

nr_subjects = numel(data.bonus);
trials_per_condition = 5;
nr_conditions = 4;

% data.consistent_with_SAT_TTB(con1,data.high_dispersion(s,:,:);
% data.consistent_with_SAT_TTB2
% data.consistent_with_SAT_TTB3
% data.consistent_with_TTB
% data.consistent_with_TTBplus
% data.consistent_with_random
for c=1:4
    
    table1(1,c) = sum(data.consistent_with_SAT_TTB(conditions(:,c)&experiment1(:)))/nr_trials_exp1;
    table1(2,c) = sum(data.consistent_with_SAT_TTB2(conditions(:,c)&experiment1(:)))/nr_trials_exp1;
    table1(3,c) = sum(data.consistent_with_SAT_TTB3(conditions(:,c)&experiment1(:)))/nr_trials_exp1;
    table1(4,c) = sum(data.consistent_with_TTB(conditions(:,c)&experiment1(:)))/nr_trials_exp1;
    table1(5,c) = sum(data.consistent_with_TTBplus(conditions(:,c)&experiment1(:)))/nr_trials_exp1;
    table1(6,c) = sum(data.consistent_with_random(conditions(:,c)&experiment1(:)))/nr_trials_exp1;
    
    table2(1,c) = sum(data.consistent_with_SAT_TTB(conditions(:,c)&experiment2(:)))/nr_trials_exp2;
    table2(2,c) = sum(data.consistent_with_SAT_TTB2(conditions(:,c)&experiment2(:)))/nr_trials_exp2;
    table2(3,c) = sum(data.consistent_with_SAT_TTB3(conditions(:,c)&experiment2(:)))/nr_trials_exp2;
    table2(4,c) = sum(data.consistent_with_TTB(conditions(:,c)&experiment2(:)))/nr_trials_exp2;
    table2(5,c) = sum(data.consistent_with_TTBplus(conditions(:,c)&experiment2(:)))/nr_trials_exp2;
    table2(6,c) = sum(data.consistent_with_random(conditions(:,c)&experiment2(:)))/nr_trials_exp2;
    
    
    
    table(1,1+c) = sum(data.consistent_with_TTB(conditions(:,c)));
    table(2,1+c) = sum(data.consistent_with_SAT_TTB(conditions(:,c)));
    table(3,1+c) = sum(data.consistent_with_random(conditions(:,c)));
    table(4,1+c) = -1;
    table(5,1+c) = sum(data.consistent_with_WADD(conditions(:,c)));
    table(6,1+c) = sum(data.consistent_with_SAT(conditions(:,c)));
end
for s=1:6
    table(s,1)=sum(table(s,2:end));
end
table

rel_freq = zeros(size(table))
rel_freq(:,2:end) = table(:,2:end) / (nr_subjects * trials_per_condition) * 100
rel_freq(:,1) = table(:,1) / (nr_subjects * nr_conditions * trials_per_condition) * 100


nr_instances_of_SAT_TTB=sum(data.consistent_with_SAT_TTB(:))
nr_instances_of_TTB=sum(data.consistent_with_TTB(:))
nr_instances_of_SAT=sum(data.consistent_with_SAT(:))


nr_instances_of_SAT_TTB_low_stakes=sum(data.consistent_with_SAT_TTB(has_low_stakes(:)))
nr_instances_of_TTB_low_stakes=sum(data.consistent_with_TTB(has_low_stakes(:)))
nr_instances_of_SAT=sum(data.consistent_with_SAT(:))

[h,p,ci,stats]=ttest2(data.pattern(has_low_stakes),data.pattern(~has_low_stakes))
%Consistent with the model's predictions we observed more alternative-based
%processing when the stakes were high than when the stakes were low
%(t(3631)=-4.26, p<0.0001). When the stakes were low there was 42% more
%outcome-based processing than alternative based processing. But when the
%stakes were high that difference fell to 32%.
[nanmean(data.pattern(~has_low_stakes)),nanmean(data.pattern(has_low_stakes))]

is_high_stakes=repmat(data.high_stakes,[1,1,10])
stats.acquisitions=[[mean(data.nr_acquisitions_low_stakes(:)),mean(data.nr_acquisitions_high_stakes(:))];...
 [sem(data.nr_acquisitions_low_stakes(:)),sem(data.nr_acquisitions_high_stakes(:))]]/28

stats.PTPROB=[[nanmean(data.percent_most_probable(~is_high_stakes(:))),...
    nanmean(data.percent_most_probable(is_high_stakes(:)))];...
 [sem(data.percent_most_probable(~is_high_stakes(:))),...
 sem(data.percent_most_probable(is_high_stakes(:)))]]

%{
stats.var_attribute=[[mean(indices_low_range.var_attribute),mean(indices_high_range.var_attribute)];...
 [sem(indices_low_range.var_attribute'),sem(indices_high_range.var_attribute')]]

stats.var_alternative=[[mean(indices_low_range.var_alternative),mean(indices_high_range.var_alternative)];...
 [sem(indices_low_range.var_alternative'),sem(indices_high_range.var_alternative')]]
%}
 
stats.pattern=[[nanmean(data.pattern(~is_high_stakes(:))),nanmean(data.pattern(is_high_stakes(:)))];...
 [sem(data.pattern(~is_high_stakes(:))),sem(data.pattern(is_high_stakes(:)))]]


stats.percent_optimal=[[mean(data.relative_performance(~is_high_stakes(:))),...
    mean(data.relative_performance(is_high_stakes(:)))];...
    [sem(data.relative_performance(~is_high_stakes(:))),...
    sem(data.relative_performance(is_high_stakes(:)))]
    ];

[h,p,ci,t_stats]=ttest2(data.relative_performance(is_high_stakes(:)),...
    data.relative_performance(~is_high_stakes(:)))

%As predicted by our model, people's relative performance was significantly
%higher in the high-stakes condition than in the low stakes condition (t(3998)=-5.89, p<0.0001).

%{
stats.percent_optimal=[[mean(indices_low_range.percent_optimal_EV),mean(indices_high_range.percent_optimal_EV)];...
 [sem(indices_low_range.percent_optimal_EV'),sem(indices_high_range.percent_optimal_EV')]]
%}
 
figure()
barwitherr([stats.acquisitions(2,:)',stats.PTPROB(2,:)',stats.percent_optimal(2,:)',stats.pattern(2,:)']',...
    [stats.acquisitions(1,:)',stats.PTPROB(1,:)',stats.percent_optimal(1,:)',stats.pattern(1,:)']')
set(gca,'FontSize',16)
legend('Low Stakes','High Stakes')
set(gca,'XTickLabel',{'% Acquisitions',...%'% Optimal',...
    '% Prioritization of argmax p(o)','Relative Performance (EV / max EV)','Alternative-based processing'},...
    'XTickLabelRotation',45)
title('Human Performance','FontSize',16)
ylabel('Percent','FontSize',16)

%% Test prediction 1: higher dispersion --> more SAT-TTB and TTB

freq_fast_and_frugal_high_dispersion = mean(or(data.consistent_with_SAT_TTB(has_high_dispersion(:)),...
data.consistent_with_TTB(has_high_dispersion(:))))

freq_fast_and_frugal_low_dispersion = mean(or(data.consistent_with_SAT_TTB(~has_high_dispersion(:)),...
data.consistent_with_TTB(~has_high_dispersion(:))))

[p,chi2,df,cohens_w] = chi2test({or(data.consistent_with_SAT_TTB(has_high_dispersion(:)),...
data.consistent_with_TTB(has_high_dispersion(:))),...
or(data.consistent_with_SAT_TTB(~has_high_dispersion(:)),...
data.consistent_with_TTB(~has_high_dispersion(:)))})

disp(['As the dispersion increased, the proportion of trials in which people relied on TTB or SAT-TTB increased from ',...
    num2str(round(100*freq_fast_and_frugal_low_dispersion),2),'% to ', num2str(round(100*freq_fast_and_frugal_high_dispersion),2),'% (chi2(1)=',num2str(chi2),', p=',num2str(p),').'])

mean(data.consistent_with_TTB(~has_high_dispersion(:)))
mean(data.consistent_with_TTB(has_high_dispersion(:)))
[p,chi2,df,cohens_w] = chi2test({data.consistent_with_TTB(has_high_dispersion(:)),...
data.consistent_with_TTB(~has_high_dispersion(:))})

mean(data.consistent_with_SAT_TTB(~has_high_dispersion(:)))
mean(data.consistent_with_SAT_TTB(has_high_dispersion(:)))
[p,chi2,df,cohens_w] = chi2test({data.consistent_with_SAT_TTB(has_high_dispersion(:)),...
data.consistent_with_SAT_TTB(~has_high_dispersion(:))})


%% Test prediction 2a: higher stakes --> more clicks
avg_nr_clicks_high_stakes = mean(data.nr_acquisitions(has_high_stakes(:)))
std_nr_clicks_high_stakes = std(data.nr_acquisitions(has_high_stakes(:)))

avg_nr_clicks_low_stakes = mean(data.nr_acquisitions(has_low_stakes(:)))
std_nr_clicks_low_stakes = std(data.nr_acquisitions(has_low_stakes(:)))

[h,p,ci,stats]=ttest2(data.nr_acquisitions(has_high_stakes(:)),...
    data.nr_acquisitions(has_low_stakes(:)))

disp(['As predicted by our resource-rational analysis, people considered a larger number of possible outcomes when the stakes were higher (',...
    num2str(round(avg_nr_clicks_low_stakes,2)),' vs. ',num2str(round(avg_nr_clicks_high_stakes,2)),...
    ', t(',num2str(stats.df),')=',num2str(round(stats.tstat),2),', p = ',num2str(p),').'])

%% Prediction 2b: When the dispersion is low, then higher stakes decrease the frequency of SAT-TTB/TTB.

ld_hs = and(has_high_stakes(:), ~has_high_dispersion(:));
ld_ls = and(has_low_stakes(:), ~has_high_dispersion(:));

data.consistent_with_FFH=or(data.consistent_with_SAT_TTB,data.consistent_with_TTB);

freq_FFH_ls=mean(data.consistent_with_FFH(has_low_stakes));
freq_FFH_hs=mean(data.consistent_with_FFH(has_high_stakes));

freq_FFH_ld_ls=mean(data.consistent_with_FFH(ld_ls));
freq_FFH_ld_hs=mean(data.consistent_with_FFH(ld_hs));

[p_2b,chi2_2b,df_2b,cohens_w_2b] = chi2test({data.consistent_with_FFH(ld_ls),...
    data.consistent_with_FFH(ld_hs)})

disp(['Consistent with the predictions of our resource-rational analysis, ',...
    'people used fast-and-frugal heuristics significantly less often when the stakes increased (',...
    num2str(round(100*freq_FFH_ls,2)),'% vs. ',num2str(round(100*freq_FFH_hs,2)), '%, chi2(1)=',...
    num2str(chi2_2b),', p=',num2str(p_2b),').'])

mean(data.consistent_with_random(~has_high_stakes))
mean(data.consistent_with_random(has_high_stakes))
[p_2b,chi2_2b,df_2b,cohens_w_2b] = chi2test({data.consistent_with_random(has_low_stakes),...
    data.consistent_with_random(has_high_stakes)})

mean(data.consistent_with_SAT_TTB(~has_high_stakes))
mean(data.consistent_with_SAT_TTB(has_high_stakes))
[p_2b,chi2_2b,df_2b,cohens_w_2b] = chi2test({data.consistent_with_SAT_TTB(has_low_stakes),...
    data.consistent_with_SAT_TTB(has_high_stakes)})

mean(data.consistent_with_TTB(~has_high_stakes))
mean(data.consistent_with_TTB(has_high_stakes))
[p_2b,chi2_2b,df_2b,cohens_w_2b] = chi2test({data.consistent_with_TTB(has_low_stakes),...
    data.consistent_with_TTB(has_high_stakes)})

mean(data.consistent_with_SAT(~has_high_stakes))
mean(data.consistent_with_SAT(has_high_stakes))
[p_2b,chi2_2b,df_2b,cohens_w_2b] = chi2test({data.consistent_with_SAT(has_low_stakes),...
    data.consistent_with_SAT(has_high_stakes)})

mean(data.consistent_with_WADD(has_low_stakes))
mean(data.consistent_with_WADD(has_high_stakes))
[p_2b,chi2_2b,df_2b,cohens_w_2b] = chi2test({data.consistent_with_WADD(has_low_stakes),...
    data.consistent_with_WADD(has_high_stakes)})

disp(['consistent_with_SAT_TTB2 (total,HS-HD,HS-LD,LS-HD,LS_LD): ',num2str(mean(data.consistent_with_SAT_TTB2(:))),' ',...
num2str(mean(data.consistent_with_SAT_TTB2(has_high_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_SAT_TTB2(has_high_stakes&has_low_dispersion))),' ',...
num2str(mean(data.consistent_with_SAT_TTB2(has_low_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_SAT_TTB2(has_low_stakes&has_low_dispersion)))])
[p,chi2,df,cohens_w] = chi2test({data.consistent_with_SAT_TTB2(has_high_stakes&has_high_dispersion),...
    [data.consistent_with_SAT_TTB2(has_high_stakes&has_low_dispersion);data.consistent_with_SAT_TTB2(has_low_stakes&has_high_dispersion);data.consistent_with_SAT_TTB2(has_low_stakes&has_low_dispersion)]})

disp(['consistent_with_TTB2plus (total,HS-HD,HS-LD,LS-HD,LS_LD): ',num2str(mean(data.consistent_with_TTB2plus(:))),' ',...
num2str(mean(data.consistent_with_TTB2plus(has_high_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_TTB2plus(has_high_stakes&has_low_dispersion))),' ',...
num2str(mean(data.consistent_with_TTB2plus(has_low_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_TTB2plus(has_low_stakes&has_low_dispersion)))])
[p,chi2,df,cohens_w] = chi2test({data.consistent_with_TTB2plus(has_high_stakes&has_high_dispersion),...
    [data.consistent_with_TTB2plus(has_high_stakes&has_low_dispersion);data.consistent_with_TTB2plus(has_low_stakes&has_high_dispersion);data.consistent_with_TTB2plus(has_low_stakes&has_low_dispersion)]})

disp(['consistent_with_SAT_TTB3 (total,HS-HD,HS-LD,LS-HD,LS_LD): ',num2str(mean(data.consistent_with_SAT_TTB3(:))),' ',...
num2str(mean(data.consistent_with_SAT_TTB3(has_high_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_SAT_TTB3(has_high_stakes&has_low_dispersion))),' ',...
num2str(mean(data.consistent_with_SAT_TTB3(has_low_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_SAT_TTB3(has_low_stakes&has_low_dispersion)))])
[p,chi2,df,cohens_w] = chi2test({data.consistent_with_SAT_TTB3(has_high_stakes&has_high_dispersion),...
    [data.consistent_with_SAT_TTB3(has_high_stakes&has_low_dispersion);data.consistent_with_SAT_TTB3(has_low_stakes&has_high_dispersion);data.consistent_with_SAT_TTB3(has_low_stakes&has_low_dispersion)]})

disp(['consistent_with_TTBplus (total,HS-HD,HS-LD,LS-HD,LS_LD): ',num2str(mean(data.consistent_with_TTBplus(:))),' ',...
num2str(mean(data.consistent_with_TTBplus(has_high_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_TTBplus(has_high_stakes&has_low_dispersion))),' ',...
num2str(mean(data.consistent_with_TTBplus(has_low_stakes&has_high_dispersion))),' ',...
num2str(mean(data.consistent_with_TTBplus(has_low_stakes&has_low_dispersion)))])
[p,chi2,df,cohens_w] = chi2test({data.consistent_with_TTBplus(has_high_stakes&has_high_dispersion),...
    [data.consistent_with_TTBplus(has_high_stakes&has_low_dispersion);data.consistent_with_TTBplus(has_low_stakes&has_high_dispersion);data.consistent_with_TTBplus(has_low_stakes&has_low_dispersion)]})

% [?TTB?, ?WADD?, ?SAT_TTB?, ?random_choice?, ?SAT_TTB2?, ?SAT?, ?TTB2_plus?, ?SAT_TTB3?, ?TTB_plus?, 'other']
strategies = [
mean(data.consistent_with_TTB(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_TTB(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_TTB(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_TTB(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_WADD(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_WADD(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_WADD(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_WADD(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_SAT_TTB(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_SAT_TTB(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_SAT_TTB(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_SAT_TTB(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_random(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_random(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_random(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_random(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_SAT_TTB2(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_SAT_TTB2(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_SAT_TTB2(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_SAT_TTB2(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_SAT(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_SAT(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_SAT(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_SAT(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_TTB2plus(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_TTB2plus(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_TTB2plus(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_TTB2plus(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_SAT_TTB3(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_SAT_TTB3(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_SAT_TTB3(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_SAT_TTB3(has_low_stakes&has_low_dispersion));
mean(data.consistent_with_TTBplus(has_high_stakes&has_high_dispersion)),mean(data.consistent_with_TTBplus(has_high_stakes&has_low_dispersion)),mean(data.consistent_with_TTBplus(has_low_stakes&has_high_dispersion)),mean(data.consistent_with_TTBplus(has_low_stakes&has_low_dispersion))];
strategies = [strategies; 1-sum(strategies(:,1)), 1-sum(strategies(:,2)), 1-sum(strategies(:,3)), 1-sum(strategies(:,4))];
%% Test prediction 2c: When the dispersion is high, then higher stakes decrease the frequency of SAT-TTB relative to the frequency of TTB.

hd_hs = and(has_high_stakes(:), has_high_dispersion(:));
hd_ls = and(has_low_stakes(:), has_high_dispersion(:));

prop_SAT_TTB = [mean(data.consistent_with_SAT_TTB(hd_ls)),mean(data.consistent_with_SAT_TTB(hd_hs))];
prop_TTB = [mean(data.consistent_with_TTB(hd_ls)),mean(data.consistent_with_TTB(hd_hs))];
ratio_SAT_TTB_to_TTB = prop_SAT_TTB./prop_TTB;

prop_TTB_of_SATTTB = prop_TTB./(prop_TTB+prop_SAT_TTB)


[p_ls,chi2_ls,df_ls,cohens_w_ls] = chi2test({data.consistent_with_SAT_TTB(hd_ls),...
    data.consistent_with_TTB(hd_ls)})

[p_hs,chi2_hs,df_ls,cohens_w_hs] = chi2test({data.consistent_with_SAT_TTB(hd_hs),...
    data.consistent_with_TTB(hd_hs)})

[prop_SAT_TTB; prop_TTB; ratio_SAT_TTB_to_TTB]

%Estimate the posterior on p_SAT_TTB_ls - p_TTB_ls
[CI_SAT_TTB_ls,samples_pSAT_TTB_ls] = proportionCI(data.consistent_with_SAT_TTB(hd_ls),0.95);
[CI_TTB_ls,samples_pTTB_ls] = proportionCI(data.consistent_with_TTB(hd_ls),0.95,1,1e9);
delta_p_ls = samples_pSAT_TTB_ls - samples_pTTB_ls;
ratio_TTB_to_SATTTB_ls =  samples_pTTB_ls ./ (samples_pSAT_TTB_ls + samples_pTTB_ls);
mean_delta_p_ls = mean(delta_p_ls);
CI_delta_p_ls = quantile(delta_p_ls,[0.025,0.975])

%Estimate the posterior on p_SAT_TTB_hs - p_TTB_hs
[CI_SAT_TTB_hs,samples_pSAT_TTB_hs] = proportionCI(data.consistent_with_SAT_TTB(hd_hs),0.95);
[CI_TTB_hs,samples_pTTB_hs] = proportionCI(data.consistent_with_TTB(hd_hs),0.95,1,1e9);
delta_p_hs = samples_pSAT_TTB_hs - samples_pTTB_hs;
ratio_TTB_to_SATTTB_hs = samples_pTTB_hs ./ (samples_pSAT_TTB_hs + samples_pTTB_hs);
mean_delta_p_hs = mean(delta_p_hs);
CI_delta_p_hs = quantile(delta_p_hs,[0.025,0.5,0.975])

%Estimate the posterior on the 

mean(ratio_TTB_to_SATTTB_hs-ratio_TTB_to_SATTTB_ls)
mean(ratio_TTB_to_SATTTB_ls),mean(ratio_TTB_to_SATTTB_hs)
CI_delta_ratio = quantile(ratio_TTB_to_SATTTB_hs-ratio_TTB_to_SATTTB_ls,[0.025,0.50,0.975])
disp(['In the high-dispersion environment increasing the stakes significantly increased the proportion of TTB sequences among all SAT-TTB  sequences from ',...
    int2str(round(100*mean(ratio_TTB_to_SATTTB_ls))),'% to ', int2str(round(100*mean(ratio_TTB_to_SATTTB_hs))),...
    '% and we can be 97.5% confident the relative frequency of TTB increased by at least ',...
    int2str(round(100*CI_delta_ratio(1),2)),'%.'])

%%
mean(data.EV_chosen_gamble_equals_maxEV(:))
mean(data.EV_chosen_gamble_equals_maxEV(has_high_dispersion(:)&has_high_stakes(:)))
mean(data.EV_chosen_gamble_equals_maxEV(has_high_dispersion(:)&has_low_stakes(:)))
mean(data.EV_chosen_gamble_equals_maxEV(has_low_dispersion(:)&has_high_stakes(:)))
mean(data.EV_chosen_gamble_equals_maxEV(has_low_dispersion(:)&has_low_stakes(:)))

nr_trial_per_condition = numel(has_high_dispersion(:))/4;
figure
hist([data.EVbest_minus_EVchosen_over_EVbest(has_high_dispersion(:)&has_high_stakes(:)),...
    data.EVbest_minus_EVchosen_over_EVbest(has_high_dispersion(:)&has_low_stakes(:)),...
    data.EVbest_minus_EVchosen_over_EVbest(has_low_dispersion(:)&has_high_stakes(:)),...
    data.EVbest_minus_EVchosen_over_EVbest(has_low_dispersion(:)&has_low_stakes(:))],10);
% h=h./sum(h);
% hist(h,10)
legend({'high-dispersion, high-stakes','high-dispersion, low-stakes','low-dispersion, high-stakes','low-dispersion, low-stakes'},'location','northwest')
xlabel('(EV(best gamble)-EV(chosen gamble))/EV(best gamble)','fontsize',18)
ylabel('frequency','fontsize',18)
title('Normalized deviation of EV of chosen gamble from best gamble','fontsize',16)
set(gca,'ylim',[0 nr_trial_per_condition],'ytick',[0:nr_trial_per_condition/4:nr_trial_per_condition],'yticklabel',0:.25:1)

figure
hist([data.EV_chosen_rank(has_high_dispersion(:)&has_high_stakes(:)),...
    data.EV_chosen_rank(has_high_dispersion(:)&has_low_stakes(:)),...
    data.EV_chosen_rank(has_low_dispersion(:)&has_high_stakes(:)),...
    data.EV_chosen_rank(has_low_dispersion(:)&has_low_stakes(:))],10);
% h=h./sum(h);
% hist(h,10)
legend({'high-dispersion, high-stakes','high-dispersion, low-stakes','low-dispersion, high-stakes','low-dispersion, low-stakes'},'location','northeast')
xlabel('rank','fontsize',18)
ylabel('frequency','fontsize',18)
title('Frequncy of chosing gamble with \{1^{st},2^{nd},...,7^{th}\} best EV','fontsize',16)
set(gca,'ylim',[0 nr_trial_per_condition],'ytick',[0:nr_trial_per_condition/4:nr_trial_per_condition],'yticklabel',0:.25:1)

% 
% data.EV_chosen_gamble_equals_maxEV(sub,b,t)
% data.EVbest_minus_EVchosen_over_EVbest(sub,b,t)
% data.EV_chosen_rank(sub,b,t)