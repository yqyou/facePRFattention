%%
clear
clc
delete(gcp('nocreate')); % shut down parpool if possible
pobj = parpool('local', 28); % open 6 cores

%% prepare the data and set the parameters
subj = 'CN056';
subfoldpath = ['/home/data/rawdata/facePRF/' subj '_faceprf/'];
cd([subfoldpath 'faceprfanalyze_vol/']);
load([subj '_PRFdatasets.mat']); % The raw data is available upon reasonable request.
res = 200;
tasks = {'face','fixation'};


%% prepare the stimulus and response for the prf model
%  RESP = GAIN*(STIM*GAU).^N
starttime = gettimestr('');

for task_i = 1:2
    n=0.2;
    stimulus = stimuli.(tasks{task_i});
    posicond = stimposicond.(tasks{task_i});
    %beta = data.beta.(tasks{task_i})(mask,:);
    beta = data.beta.(tasks{task_i});
    
    % (negative response amplitudes were set to zero)
    beta_forfitting = beta;
    beta_forfitting(beta_forfitting < 0 ) = 0;

    % prepare the prf model
    seed = [(1+res)/2 (1+res)/2 res 1 n];
    bounds = [1-res+1 1-res+1 0   -Inf 0;
                2*res-1 2*res-1 Inf  Inf Inf];

    % fitnonlinearmodel.m provides the capacity to perform stepwise fitting.
    % here, we define a version of bounds where we insert a NaN in the first
    % row in the spot that corresponds to the exponent parameter.  this 
    % indicates to fix the exponent parameter and not optimize it.
    boundsFIX = bounds;
    boundsFIX(1,5) = NaN;

    [d,xx,yy] = makegaussian2d(res,2,2,2,2);
    modelfun = @(pp,dd) pp(4)*((dd*vflatten(makegaussian2d(res,pp(1),pp(2),pp(3),pp(3),xx,yy,0,0)/(2*pi*pp(3)^2))).^pp(5));

    % stepwise fitting:
    % in the first fit, we start at the seed and optimize all
    % paramters except the exponent parameter
    % in the second fit, we start at the parameters estimated in
    % the first fit and optimize all parameters
    model = {{seed       boundsFIX   modelfun} ...
                {@(ss) ss   bounds      @(ss) modelfun}};

    resampling = 0;
    metric = @(a,b) calccod(a,b,[],[],0);

    % fit the model
    % =============== Run for loop =============================
    result = struct('params',[],'testdata',[],'modelpred',[],'trainperformance',[],'testperformance',[],'aggregatedtestperformance',[]);

    parfor ix = 1:size(beta_forfitting,1) %length(mask)
        ix
        opt = struct( ...
            'stimulus',    stimulus, ...
            'data',        beta_forfitting(ix,:)', ...
            'model',       {model}, ...
            'resampling',  resampling, ...
            'metric',      metric);
        result(ix) = fitnonlinearmodel(opt);
    end

    for ix = 1:size(beta_forfitting,1) %length(mask)
        result(ix).vn = ix;
    end

    results.(tasks{task_i}) = result;
end

endtime = gettimestr('');
disp([['Start at ' starttime],newline,['End at ' endtime]]);


% save data
save([subj '_faceprfresults.mat'],'results')

delete(gcp('nocreate')); % shut down parpool
