function Y = discretisationEigenVectorData(EigenVector)
% Y = discretisationEigenVectorData(EigenVector)
%
% discretizes previously rotated eigenvectors in discretisation
% Timothee Cour, Stella Yu, Jianbo Shi, 2004

[n,k]=size(EigenVector);


[Maximum,J]=max(EigenVector');
 
Y=sparse(1:n,J',1,n,k);     %1:n: 这是一个行向量，包含从 1 到 n 的整数
                            %J': 这是一个行向量，包含了与每个数据点对应的特征向量中的最大值相关联的索引。在这个情况下，J' 是一个长度为 n 的向量，其中的每个元素都对应于特征向量中的一个数据点，表示这个数据点在特征向量中的最大值所在的位置。
                            %1: 这个标量值是要填充到矩阵 Y 的位置上的值。在这里，我们将 1 填充到每个数据点与其最大值所在位置的交叉点上，表示这个数据点对应的离散标签。
                            %n: 这个标量表示矩阵 Y 的行数，即数据点的数量
                            %k: 这个标量表示矩阵 Y 的列数，即特征向量的数量。