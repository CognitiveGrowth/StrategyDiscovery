function [w,avg_MSE,R_total]=LinearTD0(mdp,feature_extractor,nr_episodes)
%linear TD0  algorithm that learns a linear feature-based
%approximation to the state value function for the myopic approximation to the optimal meta-level policy.
%inputs:
%  1. mdp: object of whose class implements the interface MDP
%  2. feature_extractor: a function that returns the state x features matrix
%  of state features when given a vector of states as its input
%  3. nr_episodes: number of training episodes
%  4. epsilon: probability that the epsilon-greedy policy will choose an
%  action at random

%outputs:
%  1. w: learned vector of feature weights
%  2. avg_MSE: average mean-squared error in the prediction of the state
%  value by training episode.

[s0,mdp0]=mdp.newEpisode();
w=zeros(size(feature_extractor(s0,mdp0)));
avg_MSE=zeros(nr_episodes,1);

R_total=zeros(nr_episodes,1);
nr_observations=0;
for i=1:nr_episodes    
    
    [s,mdp]=mdp.newEpisode();
    %s=s0; mdp=mdp0;
    features=feature_extractor(s,mdp);
    
    t=0; %time step within episode
    while not(mdp.isTerminalState(s))
        t=t+1;
        %1. Choose action
        [actions,mdp]=mdp.getActions(s);
        q_hat=zeros(numel(actions),1);
        VOC=NaN(numel(actions,1));
        for a=1:numel(actions)
            %[next_states,p_next_states]=mdp.predictNextState(s,actions(a));
            %next_state_values=feature_extractor(next_states)'*w;
            %q_hat(a)=mdp.expectedReward(s,actions(a))+dot(p_next_states,next_state_values);
            
            if actions(a).is_decision
                VOC(a)=0;
            else
                VOC(a)=mdp.myopicVOC(s,actions(a));
            end
        end
        PRs=mdp.getPseudoRewards(s);
        %[mdp.outcome_probabilities,PRs]
        
        if s.remaining_nr_clicks>1 %gather more information
            action=actions(argmax(VOC));
        else %choose action with highest expected value
            action.is_decision=true;
            action.gamble=argmax(s.mu);
            action.outcome=NaN;
        end
        %{
        if rand()<epsilon
            action=draw(actions);
        else
            action=actions(argmax(q_hat));
        end
        %}
        
        
        %a=sampleDiscreteDistributions(exp(q_hat)'/sum(exp(q_hat)),1);
        %action=actions(a);
        
        %2. Observe outcome
        [r,s_next]=mdp.simulateTransition(s,action);
        R_total(i)=R_total(i)+r;
        nr_observations=nr_observations+1;
        
        %3. Update weights
        alpha=1/log(2+nr_observations);
        prediction=dot(features,w);
        if mdp.isTerminalState(s_next)
            value_estimate=r;
        else
            value_estimate=r+mdp.gamma*dot(feature_extractor(s_next,mdp),w);
        end
        PE=prediction-value_estimate;
        w=w-alpha*PE*features;
        s=s_next;
        
        avg_MSE(i)=((t-1)*avg_MSE(i)+PE^2)/t;
    end 
    
    if action.is_decision
        disp(['chose gamble ',int2str(action.gamble),', MSE=',num2str(avg_MSE(i)),', |w|=',num2str(norm(w)),', return: ',num2str(R_total(i))])
    else
        disp(['inspected gamble ',int2str(action.gamble),', MSE=',num2str(avg_MSE(i)),', |w|=',num2str(norm(w)),', return: ',num2str(R_total(i))])
    end

end

end