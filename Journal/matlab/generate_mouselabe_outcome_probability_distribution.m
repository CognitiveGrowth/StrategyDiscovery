function probabilities = generate_mouselabe_outcome_probability_distribution(isHighCompensatory)

% this function will generate outcome probabilities for a single trial 
% using the same procedure used in the actual experiment
% it assumes 4 outcomes, with high-compensatory outcomes
% including one >=0.85, and low-compensatory outcomes all in the range of
% 0.1-0.4, inclusive
% isHighCompensatory is a logical indicating trial type

nr_outcomes = 4;
psumsum = 0;
cont = true;
while (psumsum ~= 1 || cont)
    cont = true;
    psumsum = 0;
    psum = 0;
    for o=1:nr_outcomes
        prob = 0;
        while (prob==0)
            prob = round(rand()*100)/100;
        end
        probabilities(o)=prob;
        psum=psum+probabilities(o);
    end
    psumsum = 0;
    for o=1:nr_outcomes
        if (probabilities(o)<0.01)
            cont = true;
        end
    end
    for o=1:nr_outcomes
        probabilities(o) = round(probabilities(o)/psum*100)/100;
        psumsum=psumsum+probabilities(o);
    end
    if isHighCompensatory
        for o=1:nr_outcomes
            if (probabilities(o)>=0.85)
                cont = false;
            end
        end
    else
        cont = false;
        for o=1:nr_outcomes
            if (probabilities(o)>=0.4 || probabilities(o)<=0.1)
                cont = true;
            end
        end
    end
end