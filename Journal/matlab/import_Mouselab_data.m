% filename=['../data/05302018/DataExp',int2str(experiment_nr),'JSON.csv'];
experiment_nr = '3conditions_300subjects';
filter_by_condition = 3;
filename=['../data/',experiment_nr,'/experiment_results.csv'];
% filename_metadata=['../data/',experiment_nr,'/MetaData.csv'];

addpath MatlabTools/parse_json/
addpath MatlabTools/

fid=fopen(filename);
nr_subjects=linecount(fid)-1+1;  % for some reason the header line isn't being read, so +1
fclose(fid);

fid=fopen(filename);
% header = fgetl(fid); % for some reason the header line isn't being read

s = 0;
for sub=1:nr_subjects
    subject_str = fgetl(fid);
%     if sub == 1
%         subject_str = subject_str(4:end);
%     end
    subject_str=strrep(regexprep(subject_str,'["]+','"'),'[]','[-999999999]');
    subject_str=regexprep(subject_str,'(?<=\:[0-9\.]+)"','');
    subject_str=regexprep(subject_str,'(?<=(\:\[[0-9\.]+)+)"','');
    subject_str=regexprep(subject_str,'(?<=\[[0-9]+)"','');
    temp=parse_json(subject_str(2:end-1));
    if filter_by_condition==1 && temp{1}.basic_info.isFullyRevealed == 0 && temp{1}.basic_info.isHiddenProbability == 0
        s=s+1;
        data_by_sub{s}=temp{1};
    elseif filter_by_condition==2 && temp{1}.basic_info.isFullyRevealed == 1 && temp{1}.basic_info.isHiddenProbability == 0
        s=s+1;
        data_by_sub{s}=temp{1};
    elseif filter_by_condition==3 && temp{1}.basic_info.isFullyRevealed == 0 && temp{1}.basic_info.isHiddenProbability == 1
        s=s+1;
        data_by_sub{s}=temp{1};
    elseif filter_by_condition==0
        s=s+1;
        data_by_sub{s}=temp{1};
    end
    disp(['Loaded data from subject ',int2str(sub)])
end
fclose(fid)

data=cellOfStructs2StructOfCells(data_by_sub)
save(['../data/',experiment_nr,'/Mouselab_data_Experiment_condition3.mat'], 'data_by_sub','data')

%{
% pay bonuses

fid=fopen(filename_metadata)
text = fileread(filename_metadata);
%worker IDs
worker_IDs=regexp(text,'[A-Z0-9]*(?=,Approved,)','match');
nr_subjects=numel(worker_IDs);
%assignment IDs
temp=regexp(text,'html",[A-Z0-9]*','match');
for s=1:numel(temp)
    assignment_IDs{s}=temp{s}(7:end);
end
%}