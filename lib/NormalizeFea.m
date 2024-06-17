function fea = NormalizeFea(fea,row)   %���õ���L2 ��һ��
%feaΪ����Ҫ��һ���ľ���row��һ���������Ϊ0�Ƕ��н��й�һ�������Ϊ1�Ƕ��н��й�һ��
if ~exist('row','var')
    row=1;
end

if row  
    Sum_feanum = size(fea,1);
    feaNorm = max(1e-14,full(sum(fea.^2,2)));% ������full() �����Ὣ��ת��Ϊ������������
    fea = spdiags(feaNorm.^-0.5,0,Sum_feanum,Sum_feanum)*fea; 
else %���н��й�һ��
    Sum_feanum = size(fea,2);
    feaNorm = max(1e-14,full(sum(fea.^2,1))');
    fea = fea*spdiags(feaNorm.^-0.5,0,Sum_feanum,Sum_feanum);
end

return
    
    
    
    

