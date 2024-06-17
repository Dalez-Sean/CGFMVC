%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function Cp = thrC(C,ro)

if (nargin < 2)
    ro = 1;
end

if (ro < 1)                    %ro���ϲ�����alpha
    N = size(C,2);             %���ؾ���C������
    Cp = zeros(N,N);
    [S,Ind] = sort(abs(C),1,'descend');%���ս������о���'abs(C)'��ÿһ�У�S����������Ind����ӦԪ����ԭ�����е�λ��
    for i = 1:N
        cL1 = sum(S(:,i));      %����S��i�еĺ�
        stop = false;
        cSum = 0; t = 0;
        while (~stop)
            t = t + 1;
            cSum = cSum + S(t,i);
            if ( cSum >= ro*cL1 )
                stop = true;
                Cp(Ind(1:t,i),i) = C(Ind(1:t,i),i);
            end
        end
    end
else
    Cp = C;
end