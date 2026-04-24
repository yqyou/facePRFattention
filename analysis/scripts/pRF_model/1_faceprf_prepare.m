clear;clc;
subj = 'CN056';
sessionnum = [1,2,3,4,5,6,7,8,9];
tasks = {'face','fixation'};

% The raw data is available upon reasonable request.
subfoldpath = ['/home/data/rawdata/facePRF/' subj '_faceprf/'];
cd([subfoldpath 'faceprfanalyze_vol/']);
for s = 1:length(sessionnum)
    sessiondir = dir(['../*session' num2str(sessionnum(s))]);
    sessionfoldpath{s} = [sessiondir.folder '/' sessiondir.name '/'];
end

%% generate contrast images
stimuli = struct();
for task = tasks
    stimposicond.(task{1}) = load(['stimposi_' task{1} '.txt']);
    stimposicond.(task{1}) = reshape(stimposicond.(task{1})',1,sum(sum(ones(size(stimposicond.(task{1}))))))';
    screen_distance = 162; 
    screen_height = 50.5; 
    Screen_size = [1920,1080];
    pix_per_degree = tan(1./2./180*pi).*screen_distance./screen_height.*2*Screen_size(2);
    face_size  = 4; % deg, diameter
    face_size = face_size*pix_per_degree;
    ecc = [-3, -1, 1, 3];  % deg
    allPosi=[];
    for j=ecc
        for i=ecc
            allPosi=[allPosi;[i,j]*pix_per_degree];
        end
    end
    res0 = 1080; % canvas
    stimposi = allPosi(stimposicond.(task{1}),:);
    contrast_images = [];
    for i = 1:length(stimposi)
        [X,Y] = meshgrid(-res0/2:1:res0/2,-res0/2:1:res0/2);
        img = (X-stimposi(i,1)).^2 + (Y-stimposi(i,2)).^2;
        img(img <= (face_size/2)^2)=1;
        img(img > (face_size/2)^2)=0;
        %figure
        %imshow(img)
        contrast_images(:,:,i) = img;
    end
    
    % downsample the images to 200x200
    res = 200;
    temp = zeros(res,res,size(contrast_images,3));
    for p = 1:size(contrast_images,3)
        temp(:,:,p) = imresize(contrast_images(:,:,p),[res,res],'cubic');
    end
    stimulus = temp;
    stimulus(stimulus < 0) = 0;
    stimulus(stimulus > 1) = 1;
    
    % reshape stimuli into a 'flattened' format: n trialsx 40000pixels
    stimuli.(task{1}) = reshape(stimulus, res^2, size(contrast_images,3))';
end


%% data--beta
% Merge the data for the same task
run_task = struct('face',[1,3,5,7,9],'fixation',[2,4,6,8,10]);
trialperrun = 32;
data.beta.face=[];
data.beta.fixation=[];
data.R2 = [];
for sess = sessionnum
    beta_all = h5read(['stats.' subj '.s' num2str(sess) '.h5'], '/data')';
    % col1-Rsquare,col2-full_Fstat,col3-coef1,col4-Tstat,col5-coef2,col6-Tstat...,colend-face_Fstat
    data.R2 = [data.R2,beta_all(:,1)]; % irrelavent to the task % m(vertices) x s(sessions)
    beta_all = beta_all(:,3:2:end-2); % m(vertices) x n(trials)
    for task = tasks

        
        if strcmp(subj,'CN044')
            if sess == 1
                rl = struct('face',[4,6],'fixation',[1,2,3,5,7]);
            elseif sess == 7
                rl = struct('face',[1,3,5,7,9,11,12],'fixation',[2,4,6,8,10]);
            elseif sess == 8
                rl = struct('face',[1,3,5,7,9,11],'fixation',[2,4,6,8,10]);
            else
                rl = run_task;
            end
            for r = rl.(task{1})
                data.beta.(task{1}) = [data.beta.(task{1}),beta_all(:,(r-1)*trialperrun+1:r*trialperrun)];
            end

        elseif strcmp(subj,'CN056')
            if sess == 1
                rl = struct('face',[1,3,5],'fixation',[2,4,6]);
            elseif sess == 5
                rl = struct('face',[2],'fixation',[1,3]);
            elseif sess == 6
                rl = struct('face',[1,3,5,7,9,11],'fixation',[2,4,6,8,10,12]);
            elseif sess == 8
                rl = struct('face',[1,3,5,7,9,11],'fixation',[2,4,6,8,10,12]);
            elseif sess == 9
                rl = struct('face',[1,3,5,7],'fixation',[2,4,6]);
            else
                rl = run_task;
            end

            for r = rl.(task{1})
                data.beta.(task{1}) = [data.beta.(task{1}),beta_all(:,(r-1)*trialperrun+1:r*trialperrun)];
            end     
            
            
        else
            for r = run_task.(task{1})
                data.beta.(task{1}) = [data.beta.(task{1}),beta_all(:,(r-1)*trialperrun+1:r*trialperrun)];
            end
        end
        
    end
end


%% save the datasets
save([subj '_PRFdatasets.mat'],'stimposicond','stimuli','data')