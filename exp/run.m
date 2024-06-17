% rewrite by Sean
%% Path setting 
clear;clc;
data_path = fullfile(pwd, "..", filesep, "data");
addpath(data_path);
lib_path = fullfile(pwd, "..", filesep, "lib");
addpath(lib_path);
code_path = fullfile(pwd, "..", filesep, "code");
addpath(genpath(code_path));

%% Datas
dirop = dir(fullfile(data_path, '*.mat'));
datas = {dirop.name};
nData = length(datas);

%% Run
for iData = 1:nData
    %*********************************************************************
    % Step 1 PreProcessing
    %*********************************************************************
    iData = datas{iData};
    iData_name = iData(1:end-4);
    fprintf("\nDataset:%s\n",iData_name);
    iData_path = fullfile(data_path,filesep,iData);
    load(iData_path);

    result = cell(1);
    result_dir = fullfile(data_path, filesep, iData_name);
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end
    best_ans_path = fullfile(result_dir,'best_ans.mat'); 
    if exist(best_ans_path,'file')
        fprintf("best_answer exists:");
        load(best_ans_path)
        fprintf("best_acc=%d best_nmi=%d\n",best_ans.eval(1,1),best_ans.eval(2,1));
        continue;
    end
  
    knn_sizes = [5,10,15,20];
    nKnn_size = length(knn_sizes);
    etas = [0,1,3,5,7,9];
    nEta = length(etas);
    nView = length(X);
    nCluster = length(unique(Y));
    issymmetric = 0;
    As = cell(nView,1);

     for iView = 1:nView
        X{iView} = sparse(double(X{iView}));
        X{iView} = NormalizeFea(X{iView}')';
        %X{iView} = NormalizeFea(X{iView});
    end

    nRepeat = 10;
    nMeasure = 13;
    evals = zeros(nMeasure+1,nRepeat);
    eval = zeros(nMeasure+1,2);
    ans = struct(...
        'evals', evals, ...
        'eval', eval, ...
        'knn_size', 0 ...
    );
    best_ans = struct(...
        'evals', evals, ...
        'eval', eval, ...
        'knn_size', 0, ...
        'eta', 0 ...
    );
    
    for iKnn_size = 1:nKnn_size
        knn_size = knn_sizes(iKnn_size);        
        ans.knn_size = knn_size;
        for iView = 1:nView
            As{iView} = constructW_PKN(X{iView}', knn_size, issymmetric);
            DA = sum(As{iView}, 2).^(-.5);
            As{iView} = sparse(As{iView});
            As{iView}(isnan(As{iView})) = 0;
            As{iView}(isinf(As{iView})) = 0;
            As{iView} = bsxfun(@times, As{iView}, DA);
            As{iView} = bsxfun(@times, As{iView}, DA');
        end
        for iEta = 1:nEta
            eta = etas(iEta);
            ans.eta = eta;
            fprintf("knn_size=%d/%d:%d, eta=%d/%d:%d\n",iKnn_size,length(knn_sizes),knn_size,iEta,length(etas),eta);
            save_filename = strcat('result_',iData_name,'_knn_size=',num2str(knn_size),'_eta=',num2str(eta),'.mat');
            result_path = fullfile(result_dir,filesep,save_filename);
            if(exist(result_path, 'file'))
               fprintf("File exists: ");
               try
                   load(result_path);
               catch ME
                   warning('The file loads error%s: %s', result_path, ME.message);
                   if ~exist(fullfile(result_dir,filesep,'load_error'),'dir')
                       mkdir(fullfile(result_dir,filesep,'load_error'));
                   end
                   save(fullfile(result_dir,filesep,'load_error',filesep,save_filename), 'lambda');
               end
            else
                seed = 2024;
                rng(seed,'twister');
                random_seeds = randi([0, 1000000], 1, nRepeat);
                for iRepeat = 1:nRepeat
                    rng(random_seeds(iRepeat),'twister');
                    tic;    
                    [label, alpha, beta, objHistory] = SMGC(As, nCluster, eta, knn_size);
                    ans.evals(nMeasure+1,iRepeat) = toc;
                    ans.evals(1:nMeasure,iRepeat) = my_eval_y(label, Y);
                end
                
                for iMeasure = 1:nMeasure+1
                    ans.eval(iMeasure,1) = mean(ans.evals(iMeasure,:));
                    ans.eval(iMeasure,2) = std(ans.evals(iMeasure,:));
                end
                if(ans.eval(1,1)>best_ans.eval(1,1))
                    best_ans = ans;
                end
                save(result_path, 'ans');
                fprintf("SingleFile is saved: ");    
            end
            fprintf("acc=%d nmi=%d\n",ans.eval(1,1),ans.eval(2,1));
        end  
    end
    save(fullfile(result_dir,'best_ans.mat'), 'best_ans');
    fprintf("BestFile is saved: ");
    fprintf("best_acc=%d best_nmi=%d\n",best_ans.eval(1,1),best_ans.eval(2,1));
end
