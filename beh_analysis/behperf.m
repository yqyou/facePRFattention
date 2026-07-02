%% set parameters
close all; clear all; clc;
subj_list= {'CN040','CN041','CN042','CN043','CN044','CN045','CN055','CN056'};

%% read data
for sid = 1:length(subj_list)
    subj = subj_list{sid};

    % The raw data is available upon reasonable request.
    session_list = dir(['/home/data/rawdata/facePRF/' subj '_faceprf/*session*']);

    for s = 1:length(session_list)
        if isempty(regexp(session_list(s).name,'session[1-9]'))
            session_list(s).name = [];
        end
    end

    % calculate behperf
    behperf = struct('pH',[],'pF',[],'dprime',[]);
    for s = 1:length(session_list)
        if ~isempty(session_list(s).name)
            sess = str2num(session_list(s).name(end));
            file_list = dir([session_list(s).folder '/' session_list(s).name '/mat_files_from_exp/*mat']);
            for ff = 1:length(file_list)
                load([file_list(ff).folder '/' file_list(ff).name]);
                [behperf.pH(sess,ff), behperf.pF(sess,ff), behperf.dprime(sess,ff)]=checkresponse(result);
            end
        end
    end
    behperf_all(sid) = behperf;
end
%%
% remove CN044 session1 run1,3,5
behperf_all(5).pH(1,[1,3,5])=[behperf_all(5).pH(7,[11,12]),behperf_all(5).pH(8,11)];
behperf_all(5).pF(1,[1,3,5])=[behperf_all(5).pF(7,[11,12]),behperf_all(5).pF(8,11)];
behperf_all(5).dprime(1,[1,3,5])=[behperf_all(5).dprime(7,[11,12]),behperf_all(5).dprime(8,11)];
behperf_all(5).pH(:,[11,12])=[];
behperf_all(5).pF(:,[11,12])=[];
behperf_all(5).dprime(:,[11,12])=[];
% replace CN056 session1&5
behperf_all(8).pH(1,[7:10]) = [behperf_all(8).pH(8,[11,12]),behperf_all(8).pH(6,[11,12])];
behperf_all(8).pF(1,[7:10]) = [behperf_all(8).pF(8,[11,12]),behperf_all(8).pF(6,[11,12])];
behperf_all(8).dprime(1,[7:10]) = [behperf_all(8).dprime(8,[11,12]),behperf_all(8).dprime(6,[11,12])];
behperf_all(8).pH(5,[1:10]) = [behperf_all(8).pH(9,1:7),behperf_all(8).pH(5,1:3)];
behperf_all(8).pF(5,[1:10]) = [behperf_all(8).pF(9,1:7),behperf_all(8).pF(5,1:3)];
behperf_all(8).dprime(5,[1:10]) = [behperf_all(8).dprime(9,1:7),behperf_all(8).dprime(5,1:3)];
behperf_all(8).pH(:,[11,12])=[];
behperf_all(8).pF(:,[11,12])=[];
behperf_all(8).dprime(:,[11,12])=[];
behperf_all(8).pH(9,:)=[];
behperf_all(8).pF(9,:)=[];
behperf_all(8).dprime(9,:)=[];

for sid = 1:length(subj_list)
    behperf_all(sid).acc = (behperf_all(sid).pH+1-behperf_all(sid).pF)/2;
end

save('behperf_faceprf.mat','behperf_all');


%% function
function [hit_rate,fa_rate,dprime]=checkresponse(result)
% check the resp and task

    % task
    if  strfind(result.exp_type,'face')
        % number of trials (signal+noise)
        trial_num = length(find(result.this_face_list))/2*3/4; % 3 trials in a sequence
        % signal(repeated face) list
        face_list = floor((result.this_face_list-1)/7)+1; % face identity
        face_list = face_list(1:2:end);
        signal_ind = find(face_list(2:end)&diff(face_list)==0)*2+1; % repeat stimulus but not 0
        signal_starttime = (signal_ind-1)*0.5;  % frame to second
    elseif strfind(result.exp_type,'fixation')
        % number of trials (signal+noise)
        trial_num = length(result.this_digit_list);
        % signal(repeated digit) list
        digit_list = result.this_digit_list;
        signal_ind = find(diff(digit_list)==0)+1; % frame
        signal_starttime = (signal_ind-1)*0.5;  % frame to second
    end

    % signal&noise trial
    signal_num = length(signal_starttime);
    noise_num = trial_num - signal_num;

    % response keys
    resp_time = result.key_record(2,find(result.key_record(1,:))); % second
    resp_time = resp_time(2:end); % delete the initial key

    % hit, miss; false alarm, correct reject
    hit = 0;
    for s = 1:signal_num
        resp_window = [signal_starttime(s),signal_starttime(s)+2]; % 2 second after repeated stimulus for response
        resplist = resp_time((resp_time >= resp_window(1))&(resp_time <= resp_window(2)));
        if ~isempty(resplist)
            hit = hit+1;
            resp_time(resp_time==resplist(1)) = []; % only count once per response
        end
    end
    hit_rate = hit/signal_num;
    fa_rate = length(resp_time)/noise_num;
    dprime = norminv(min(hit_rate,1-1/signal_num),0,1) - norminv(max(fa_rate,1/noise_num),0,1);
    % to ensure finite d' values for those cases in which there was no false alarms or no misses
end