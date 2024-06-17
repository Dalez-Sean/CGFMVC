function [label, alpha, beta, objHistory] = HKLFMVC(As, nCluster, eta, knn_size)
nView = length(As);
nSmp = size(As{1}, 1);
HLAs = cell(nView, nView);
HLs = cell(nView,1);
for iView1 = 1:nView
    L = speye(nSmp) - As{iView1};
    L = (L + L')/2;
    HLs{iView1} = expm(-eta * L);
    for iView2 = 1:nView
        HLAs{iView1, iView2} = HLs{iView1} * As{iView2};
    end
end

%*************************************************
% Initialization of alpha, beta, gamma
%*************************************************
beta = ones(nView, 1)/nView;
alpha = ones(nView, 1)/nView;
gamma = 1;

%*************************************************
% Initialization of S
%*************************************************
B = zeros(nSmp);
for iView1 = 1:nView
    tmp = zeros(nSmp);
    for iView2 = 1:nView
        temp = alpha(iView1) * HLAs{iView2,iView1};
    end
    B = (1/beta(iView1)) * temp + B;
end
B = B/sum(1./beta);
B = B - 1e8*eye(nSmp);
[~, Idx] = sort(B, 2, 'descend');
Idx = Idx(:, 1:knn_size);
S = zeros(nSmp);
for iSmp = 1:nSmp
    idxa0 = Idx(iSmp, :);
    ad = B(iSmp, idxa0);
    S(iSmp, idxa0) = EProjSimplex_new(ad);
end

%********************************************
% Initialization of H
%********************************************
S2 = S + S';
L = diag(sum(S2,2)) - S2;
L = (L + L')/2;
[H, ~] = eigs(L, nCluster, 'SA');

iter = 0;
objHistory = [];
converges = false;
maxIter = 50;
while ~converges
    
    %*******************************************
    % Update S
    %*******************************************
    B = zeros(nSmp);
    for iView1 = 1:nView
        tmp = zeros(nSmp);
        for iView2 = 1:nView
            tmp = tmp + alpha(iView2) * HLAs{iView2,iView1};
        end
        B = (1/beta(iView1)) * tmp + B;
    end
    DH = EuDist2(H, H, 0);
    BDH = B - gamma * DH;
    BDH = BDH/sum(1./beta) ;
    BDH = BDH - 1e8*eye(nSmp);
    [~, Idx] = sort(BDH, 2, 'descend');
    Idx = Idx(:, 1:knn_size);
    S = zeros(nSmp);
    for iSmp = 1:nSmp
        idxa0 = Idx(iSmp, :);
        ad = BDH(iSmp, idxa0);
        S(iSmp, idxa0) = EProjSimplex_new(ad);
    end
    
    %***********************************************
    % Update alpha
    %***********************************************
    A = zeros(nView, nView);
    b = zeros(nView, 1);
    for iView1 = 1 : nView
        for iView2 = 1 : nView
            LHW1 = HLAs{iView2, iView1} ;
            for iView3 = 1 : nView
                LHW2 = HLAs{iView3, iView1};
                A(iView2, iView3) = A(iView2, iView3) + beta(iView1) * sum(sum( LHW1 .* LHW2));
            end
            b(iView1) = b(iView1) + beta(iView1) * sum(sum( LHW1 .* S));
        end
    end
    opt = [];
    opt.Display = 'off';
    alpha = quadprog(A, -b, [], [], ones(1, nView), 1, zeros(nView, 1), ones(nView, 1), [], opt);
    
    %***********************************************
    % Update beta
    %***********************************************
    es = zeros(nView, 1);
    for iView1 = 1 : nView
        tmp = zeros(nSmp, nSmp);
        for iView2 = 1:nView
            tmp = tmp + alpha(iView2) * HLAs{iView2, iView1};
        end
        E = tmp - S;
        es(iView1) = sum(sum( E.^2 ));
    end
    beta = sqrt(es)/sum(sqrt(es));
    
    obj = sum(beta .* es);
    objHistory = [objHistory; obj]; %#ok

    if iter > 2 && (abs(objHistory(iter-1)-objHistory(iter))<0)
        iter
        converges = 0
    end
    if iter > 2 && (abs((objHistory(iter-1)-objHistory(iter))/objHistory(iter-1))<1e-6)
        converges = 1;
    end
    
    if iter > maxIter
        converges = 1;
    end
    iter = iter + 1;
end
CKSym = (S + S')/2;
CKSym = BuildAdjacency(thrC(CKSym, 0.7));
H_normalized = SpectralClustering_ncut(CKSym, nCluster);
seed = 2024;
rng(seed,"twister");
label = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
end

