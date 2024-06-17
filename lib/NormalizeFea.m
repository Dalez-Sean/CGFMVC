function fea = NormalizeFea(fea,row)   %采用的是L2 归一化
%fea为输入要归一化的矩阵，row是一个数，如果为0是对列进行归一化，如果为1是对行进行归一化
if ~exist('row','var')
    row=1;
end

if row  
    Sum_feanum = size(fea,1);
    feaNorm = max(1e-14,full(sum(fea.^2,2)));% 在这里full() 函数会将其转换为完整的列向量
    fea = spdiags(feaNorm.^-0.5,0,Sum_feanum,Sum_feanum)*fea; 
else %对列进行归一化
    Sum_feanum = size(fea,2);
    feaNorm = max(1e-14,full(sum(fea.^2,1))');
    fea = fea*spdiags(feaNorm.^-0.5,0,Sum_feanum,Sum_feanum);
end

return
    
    
    
    

