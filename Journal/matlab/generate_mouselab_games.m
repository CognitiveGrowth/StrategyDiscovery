function [mouselab_matrix, probabilities] = generate_mouselab_games(nr_games)

% generate mouselab games to save and sample from in

if rem(nr_games,4)
    error('Number of games needs to be a multiple of 4 for 4 conditions')
end

payoff_range1 = [0.01, 0.25];
payoff_range2 = [0.01, 9.99];

nr_gambles = 7;
nr_outcomes = 4;

for game = 1:nr_games
    if game <= nr_games/2
        payoff_range = payoff_range1;
    else
        payoff_range = payoff_range2;
    end
    payoff_mu = (payoff_range(1)+payoff_range(2))/2;
    payoff_std = 0.3*(payoff_range(1)-payoff_range(2));
    i = 0;
    for g = 1:nr_gambles
        for o = 1:nr_outcomes
            i = i+1;
            mouselab_matrix(i,game) = round(randn_trunc(payoff_mu,payoff_std,payoff_range),2);
            
        end
    end
    if game <= nr_games/4 || (game > nr_games/2 && game <= nr_games*3/4)
        probabilities(:,game) = generate_mouselab_outcome_probability_distribution(1);
    else
        probabilities(:,game) = generate_mouselab_outcome_probability_distribution(0);
    end
end



function randnn = randn_trunc(mu,sd,range)
ii = 1;
while (ii == 1 || randnn < range(1) || randnn > range(2))
    ii = 0;
    u = 1 - rand(); % Subtraction to flip [0, 1) to (0, 1].
    v = 1 - rand();
    randnn = sqrt( -2.0 * log( u ) ) * cos( 2.0 * pi * v );
    randnn = randnn*sd+mu;
end