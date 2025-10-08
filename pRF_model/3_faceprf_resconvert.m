clear;clc

subjs = {'CN040','CN041','CN042','CN043','CN044','CN045','CN055','CN056'};
tasks = {'face','fixation'};
roi_labels = {'V1','V2','V3','hV4','IOG','pFus','mFus'};

% set unit
screen_distance=162;
screen_height=50.5;
Screen_size=[1920,1080];
pix_per_degree=tan(1./2./180*pi).*screen_distance./screen_height.*2*Screen_size(2);
res0 = 1080;
res = 200;
cfactor = res0/res/pix_per_degree; % downsampled stimulus pixel to visual degree
exptlowerbound = 0.001;


%% faceprfres: calculate ecc, size, gain    unit:deg
face_size  = 4; % deg, diameter
face_size = face_size*pix_per_degree;
ecc = [-3, -1, 1, 3];  % deg
allPosi=[];
for j=ecc
    for i=ecc
        allPosi=[allPosi;[i,j]*pix_per_degree];
    end
end
contrast_images = [];
for i = 1:length(allPosi)
    [X,Y] = meshgrid(-res0/2:1:res0/2,-res0/2:1:res0/2);
    img = (X-allPosi(i,1)).^2 + (Y-allPosi(i,2)).^2;
    img(img <= (face_size/2)^2)=1;
    img(img > (face_size/2)^2)=0;
    contrast_images(:,:,i) = img;
end
temp = zeros(res,res,size(contrast_images,3));
for p = 1:size(contrast_images,3)
    temp(:,:,p) = imresize(contrast_images(:,:,p),[res,res],'cubic');
end
% ensure that all values are 0 or 1
temp(temp < 0) = 0;
temp(temp > 1) = 1;
stimuli = reshape(temp, res^2, size(contrast_images,3))';

for subj_i = 1:8
    subj = subjs{subj_i};
    cd(['/home/data/rawdata/facePRF/' subj '_faceprf/faceprfanalyze_vol'])

    %%
    % 1-vind, 2-R2, 3-x, 4-y, 5-ecc, 6-ang, 7-size, 8-expt, 9-gain
    load([ subj '_faceprfresults.mat']);% The raw data is available upon reasonable request.
    faceprfres = struct();
    for task_i = 1:2
        result = results.(tasks{task_i});
        params = reshape([result.params],5,length(result))';
        vind = [result.vn]';
        R2 = [result.trainperformance]';
        x = (params(:,2) - (1+res)/2)*cfactor;
        y = ((1+res)/2 - params(:,1))*cfactor;
        ecc = sqrt(x.^2 + y.^2);
        ang = mod(atan2(y,x),2*pi)/pi*180;
        ang(ecc==0) = NaN; % check for special case for ecc==0; set angle to NaN since it is undefined
        size = abs(params(:,3))./sqrt(posrect(params(:,5),exptlowerbound))*cfactor; % sigma/sqrt(n)
        expt = posrect(params(:,5),exptlowerbound);
        gain_params = params(:,4);
        gain = [];
        for vi = 1:length(result)
            pp = params(vi,:);
            gain(vi,1) = max(pp(4)*((stimuli*vflatten(makegaussian2d(res,pp(1),pp(2),pp(3),pp(3))/(2*pi*pp(3)^2))).^pp(5)));
        end
        faceprfres.(tasks{task_i}) = [vind,R2,x,y,ecc,ang,size,expt,gain];        
        % quicker access to results
        prfres = struct();
        for task_i = 1:2
            result = results.(tasks{task_i});
            params = reshape([result.params],5,length(result))';
            vind = [result.vn]';
            R2 = [result.trainperformance]';
            prfres.(tasks{task_i}) = [vind,R2,params];
        end        
    end
    save([ subj '_faceprfresults.mat'],'results','prfres','faceprfres');
end
