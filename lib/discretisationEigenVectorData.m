function Y = discretisationEigenVectorData(EigenVector)
% Y = discretisationEigenVectorData(EigenVector)
%
% discretizes previously rotated eigenvectors in discretisation
% Timothee Cour, Stella Yu, Jianbo Shi, 2004

[n,k]=size(EigenVector);


[Maximum,J]=max(EigenVector');
 
Y=sparse(1:n,J',1,n,k);     %1:n: ����һ���������������� 1 �� n ������
                            %J': ����һ������������������ÿ�����ݵ��Ӧ�����������е����ֵ����������������������£�J' ��һ������Ϊ n �����������е�ÿ��Ԫ�ض���Ӧ�����������е�һ�����ݵ㣬��ʾ������ݵ������������е����ֵ���ڵ�λ�á�
                            %1: �������ֵ��Ҫ��䵽���� Y ��λ���ϵ�ֵ����������ǽ� 1 ��䵽ÿ�����ݵ��������ֵ����λ�õĽ�����ϣ���ʾ������ݵ��Ӧ����ɢ��ǩ��
                            %n: ���������ʾ���� Y �������������ݵ������
                            %k: ���������ʾ���� Y ��������������������������