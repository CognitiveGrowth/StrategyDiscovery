freq_chosen_gamble_is_max_EV = [];
for s = size(data.EVs,1)
    if data.condition(s) == 2
        for b = 1:size(data.EVs,2)
            for t = 1:size(data.EVs,3)
                freq_chosen_gamble_is_max_EV = [freq_chosen_gamble_is_max_EV,(data.EV_chosen_gamble(s,b,t) == max(squeeze(data.EVs(s,b,t,:))))];
            end
        end
    end
end