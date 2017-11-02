function [R_total,problems,states,choices,indices,has_high_dispersion]=...
    inspectPolicy(mdp,feature_extractor,glm,epsilon,nr_episodes)
%inspect a policy that is based on a linear feature-based approximation
%to the state value function for the epsilon-greedy policy.
%inputs:
%  1. mdp: object of whose class implements the interface MDP
%  2. feature_extractor: a function that returns the state x features matrix
%  of state features when given a vector of states as its input
%  3. nr_episodes: number of training episodes

%outputs:

has_high_dispersion=NaN(nr_episodes,1);

[~,mdp0]=mdp.newEpisode();
actions=mdp0.actions;

R_total=zeros(nr_episodes,1);
nr_observations=0;
for i=1:nr_episodes

    [s,mdp]=mdp.newEpisode();
    
    problems(:,:,i)=[mdp.outcome_probabilities,mdp.payoff_matrix];
    choices(:,:,i)=NaN(mdp.nr_outcomes+1,mdp.nr_gambles);
    
    has_high_dispersion(i)=max(mdp.outcome_probabilities)>0.8;
    
    t=0; %time step within episode
    nr_alternative_based_transitions=0;
    nr_attribute_based_transitions=0;
    prev_choice=[NaN,NaN];
    while not(mdp.isTerminalState(s))
        t=t+1;
        
        states{i}(t)=s;
                
        %1. Choose action
        [actions,mdp]=mdp.getActions(s);
        action=contextualThompsonSampling(s,mdp,glm);
        
        if action.is_decision
            choices(1,action.gamble,i)=t;
        else
            choices(1+action.outcome,action.gamble,i)=t;
            
            if and(action.outcome==prev_choice(1),action.gamble~=prev_choice(2))
                nr_attribute_based_transitions=nr_attribute_based_transitions+1;
            elseif and(action.outcome~=prev_choice(1),action.gamble==prev_choice(2))
                nr_alternative_based_transitions=nr_alternative_based_transitions+1;
            end
            
            prev_choice=[action.outcome,action.gamble];
        end
            
        %2. Observe outcome
        [r,s_next,PR]=mdp.simulateTransition(s,action);
        R_total(i)=R_total(i)+r;
        nr_observations=nr_observations+1;
        
        s=s_next;
        
    end
    
    %EV(choice)/[max_g EV(g)]
    EVs=mdp.payoff_matrix'*mdp.outcome_probabilities(:);
    max_EV(i)=max(mdp.payoff_matrix'*mdp.outcome_probabilities(:));
    EV_of_choice(i)=EVs(action.gamble);
    indices.percent_optimal_EV(i)=100*EV_of_choice(i)/max_EV(i);
    
    %analyze difference between compensatory vs. non-compensatory problems
    indices.was_compensatory(i)=max(mdp.outcome_probabilities)<=0.40;
    
    
    %number of acquisitions
    indices.nr_acquisitions(i)=sum(not(isnan(s.observations(:))));
    
    %proportion of time spent on the Most Important Attribute (MIA)
    MIA=argmax(mdp.outcome_probabilities);
    indices.PTPROB(i)=sum(not(isnan(s.observations(MIA,:))))/sum(not(isnan(s.observations(:))));
        
    %variance across attributes
    nr_acquisitions_by_attribute=sum(not(isnan(s.observations)),2);
    indices.var_attribute(i)=var(nr_acquisitions_by_attribute);
    
    %variance accross alternatives
    nr_acquisitions_by_alternative=sum(not(isnan(s.observations)),1);
    indices.var_alternative(i)=var(nr_acquisitions_by_alternative);
    
    %relative frequency of alternative-based processing
    indices.pattern(i)=(nr_alternative_based_transitions-nr_attribute_based_transitions)/...
        (nr_alternative_based_transitions+nr_attribute_based_transitions);
    
    %disp(['MSE=',num2str(avg_MSE(i)),', |w|=',num2str(norm(w)),', return: ',num2str(R_total(i))])
end

indices.effect_of_dispersion.means=[nanmean(indices.PTPROB(~indices.was_compensatory)),
    nanmean(indices.PTPROB(indices.was_compensatory))];
[h,p,ci,stats]=ttest2(indices.PTPROB(~indices.was_compensatory),indices.PTPROB(indices.was_compensatory));
indices.effect_of_dispersion.stats=stats;
indices.effect_of_dispersion.p=p;

end