function  D = EuDist(fea_a,fea_b,bSqrt)
% bSqrt���������һ���߼�ֵ�����ǽ���������ľ��뿪���ţ���ֵĬ��Ϊtrue
if ~exist('bSqrt','var')
    bSqrt = 1;  %�� MATLAB �У��߼�ֵ true ͨ�������� 1 ����ʾ�����߼�ֵ false ͨ�������� 0 ����ʾ��
end

if  (~exist('fea_b','var'))||isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@pius,aa,aa') -2*ab; %bsxfun��һ������������������Ԫ���ۼӣ�@plus�Ǻ����������ʾ�ӷ�����
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D'); % ��֤D�ǶԳƾ���ȡ�������ǻ������ǲ��ֲ���ֵ����һ�룬ʹ��Գ�
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
        
    
