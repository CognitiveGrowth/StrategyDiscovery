%addpath('~/Dropbox/PhD/MatlabTools/')
%addpath('~/Dropbox/PhD/MatlabTools/parse_json/')
%filename_metadata = '~/Dropbox/mouselab_cogsci17/data/mouselab_cogsci17_metadata.csv';

datdir = 'fullyRevealed_100uniqueTrials';
load(['../data/',datdir,'/Mouselab_data_Experiment.mat']);

for s = 1:length(data_by_sub)
    if ~isfield(data_by_sub{s},'bonus')
    
        %         bonus(s) = 3.53;
    else
        bonus(s) = str2num(data_by_sub{s}.bonus);
    end
end

% text=fileread(['../data/pilot_3conditions/worker_IDs.csv']);

% worker_IDs=text;%regexp(text,'[A-Z0-9]*(?=,Approved,)','match');

% temp=regexp(text,'(?<=\=3WJGKMRWVI9L0ARR9VU4U5TG12DCDY,)[A-Z0-9]*','match');
% for s=1:numel(temp)
%     assignment_IDs{s}=temp{s};
% end

% fid=fopen(['../data/pilot_3conditions/worker_IDs.csv']);
% nr_subjects=linecount(fid)-1+1
% for sub=1:nr_subjects
%     worker_IDs{sub} = fgetl(fid);
% end

bonus_file='';
for s=1:numel(worker_IDs)
    
    bonus_commands{s}=['./grantBonus.sh -workerid ',worker_IDs{s},...
        ' -assignment ', assignment_IDs{s}, ' -amount ', num2str(bonus(s)),...
        ' -reason "Bonus in Betting Game."'];
    bonus_file=[bonus_file, bonus_commands{s},';'];
end

filename=['payBonuses_',datdir,'.sh'];
unix(['rm ',filename])
fid = fopen(filename, 'w');
% print a title, followed by a blank line
fprintf(fid, bonus_file);
fclose(fid)
unix(['chmod u+x ',filename])
unix(['mv ',filename,' ../data/',datdir,'/',filename])

