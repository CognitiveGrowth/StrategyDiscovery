function [ep,out] = VBA_groupBMCbtw(L,options,factors)
% group-level between-condition Bayesian model comparison
% function Ltilde = VBA_BTWgroup(L,options)
% IN:
%   - L: nmXnsxnc array of log-model evidences (nm models; ns subjects; nc
%   conditions).
%   - options: a structure containing the following fields:
%       .DisplayWin: flag for display window
%       .verbose: flag for summary statistics display
%       .families: a cell array of size nf, which contains the indices of
%       the models that belong to each of the nf families.
%   - factors: n1Xn2X...Xnf factorial condition attribution matrix. Each
%   entry contains the index of the corresponding condition. This is used in
%   case of a factorial design, to gather conditions into orthogonal
%   factors.
% OUT:
%   - ep: a nfX1 vector containing the exceedance probability of no
%   difference in models along each of the nf dimensions of the factorial
%   design space (if design is not factorial, then nf=1).
%   - out: structure containing the following fields:
%       .dt: the algorithm execution time (in sec)
%       .date: date vector, in matlab format (see clock.m)
%       .families: a cell array of size 2Xnf, which contains the indices of
%       the models that belong to each of the tuples families 'equal' and
%       'not equal', for each dimension of the factorial design.
%       .options: this is useful when using default options
%       .C: ncXnt tuple label matrix (where the number of tuples is
%       nt=nm^nc). Each column identifies a nc-tuple by the index of models
%       in each of the  possible conditions.
%       .VBA: a nfX1 structure containing the output of VBA_groupBMC.m,
%       which has been called for each dimension of the factorial design.

options.tStart = tic;
[nm,ns,nc] = size(L);
try;options;catch;options=[];end
if nc==1
    disp('Error: only one condition detected!')
    ep = [];
    out = [];
    return
end
if ~isfield(options,'verbose')
    options.verbose = 1;
end
if ~isfield(options,'DisplayWin')
    options.DisplayWin = 0;
end
if ~isfield(options,'families')
    options.families = [];
end
if ~isempty(options.families)
    nfam = length(options.families);
    Cfam = zeros(nm,1);
    tmp = [];
    for i=1:nfam
        indf = options.families{i};
        if isempty(indf)
            disp(['Error: family #',num2str(i),' is empty!'])
            ep = [];
            out = [];
            return
        end
        Cfam(indf) = i;
        if ~isempty(intersect(tmp,indf))
            disp('Error: families are not mutually exclusive!')
            ep = [];
            out = [];
            return
        end
        tmp = [tmp;vec(indf)];
    end
    if ~isequal(vec(unique(tmp)),vec(1:nm))
        if numel(unique(tmp)) < nm
            disp('Error: families do not cover the entire set of models!')
        else
            disp('Error: families contain models that do not exist!')
        end
        ep = [];
        out = [];
        return
    end    
else
    Cfam = vec(1:nm);
end


VBA_disp(['Date: ',datestr(date)],options)
try;factors;catch;factors=vec(1:nc);end
sf = size(factors);
sf(sf<=1) = [];
nf = size(sf,2); % number of factors/dimensions across conditions
indf = cell(nf,1);
for f=1:nf
    nlevels = size(factors,f); % #levels in this dimension
    nconds = numel(factors)/nlevels; % #conditions per level
    indf{f} = zeros(nconds,nlevels);
    strpar = repmat([':,'],1,nf);
    strpar(end) = [];
    strpar(2*(f-1)+1) = 'k';
    for k=1:nlevels
        eval(['indf{f}(:,k) = vec(factors(',strpar,'));'])
    end
end
% form nc-tuples
[Ccon] = VBA_getNtuples(nm,nc,0);
nt = size(Ccon,2);
fam = cell(2,nf);
Lt = zeros(nt,ns);
if options.verbose
    fprintf(1,['Forming tuples families... ']);
    fprintf(1,'%6.2f %%',0)
end
for i=1:nt
    Ci = Ccon(:,i);
    for j=1:nc
        Lt(i,:) = Lt(i,:) + L(Ci(j),:,j);
    end
    % pool conditions across dimension
    for f=1:nf
        nconds = size(indf{f},1);
        flag = 1;
        for k=1:nconds
            flag = flag && isequal(length(unique(Cfam(Ci(indf{f}(k,:))))),1);
        end
        if flag
            fam{1,f} = [fam{1,f};i];
        else
            fam{2,f} = [fam{2,f};i];
        end
    end
    if options.verbose
        fprintf(1,repmat('\b',1,8))
        fprintf(1,'%6.2f %%',100*i/nt)
    end
end
if options.verbose
    fprintf(1,repmat('\b',1,8))
    fprintf(' OK.')
    fprintf('\n')
end

opt.verbose = 0;
opt.DisplayWin = options.DisplayWin;
% opt.priors.a = ones(nt,1);
ep = zeros(nf,1);
out.pep = zeros(nf,1);
for f=1:nf
    if options.verbose && nf >1
        disp('---')
        disp(['[factorial design: dimension #',num2str(f),' (',num2str(size(factors,f)),' levels)]'])
        disp('---')
    end
    opt.families = fam(:,f);
    [out.VBA(f).posterior,out.VBA(f).out] = VBA_groupBMC(Lt,opt);
    
    %-- display summary statistics
    ep(f) = out.VBA(f).out.families.ep(1);
    out.pep(f) = ep(f).*(1-out.VBA(f).out.bor) + 0.5*out.VBA(f).out.bor;
    
    if options.verbose
        if floor(out.VBA(f).out.dt./60) == 0
            timeString = [num2str(floor(out.VBA(f).out.dt)),' sec'];
        else
            timeString = [num2str(floor(out.VBA(f).out.dt./60)),' min'];
        end
        fprintf(['VB converged in ',num2str(length(out.VBA(f).out.F)),' iterations (took ~',timeString,').','\n'])
        fprintf(['Dimensions:','\n'])
        fprintf(['     - subjects: n=',num2str(ns),'\n'])
        fprintf(['     - conditions: c=',num2str(nc),'\n'])
        fprintf(['     - models: K=',num2str(nm),'\n'])
        fprintf(['     - ',num2str(nc),'-tuples: t=',num2str(nt),'\n'])
        fprintf(['Posterior probabilities:','\n'])
        fprintf(['     - RFX: p(H1|y)= ','%4.3f','\n'],1-out.VBA(f).out.bor)
        fprintf(['     - null: p(H0|y)= ','%4.3f','\n'],out.VBA(f).out.bor)
        fprintf(['Assessment of model stability across conditions:','\n'])
        fprintf(['     - exc. prob.: ','%4.3f','\n'],ep(f))
        fprintf(['     - protected exc. prob.: ','%4.3f','\n'],out.pep(f))
        fprintf('\n')
    end
end

% wrap-up
out.dt = toc(options.tStart);
out.date = clock;
out.options = options;
out.families = fam;
out.Ccon = Ccon;



