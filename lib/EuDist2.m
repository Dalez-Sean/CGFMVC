function  D = EuDist(fea_a,fea_b,bSqrt)
% bSqrt这个参数是一个逻辑值，它是将计算出来的距离开根号，其值默认为true
if ~exist('bSqrt','var')
    bSqrt = 1;  %在 MATLAB 中，逻辑值 true 通常用整数 1 来表示，而逻辑值 false 通常用整数 0 来表示。
end

if  (~exist('fea_b','var'))||isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@pius,aa,aa') -2*ab; %bsxfun是一个操作函数，进行逐元素累加，@plus是函数句柄，表示加法操作
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D'); % 保证D是对称矩阵，取其上三角或下三角部分并赋值给另一半，使其对称
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';
    
    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end
    
    D = bsxfun(@plus,aa,bb')- 2*ab;
    D(D<0) =0;
    if bSqrt 
        D =sqrt(D);
    end
end
        
    
