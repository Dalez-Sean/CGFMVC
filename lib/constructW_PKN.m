% construct similarity matrix with probabilistic k-nearest neighbors. It is a parameter free, distance consistent similarity.
function A = constructW_PKN(X, k, issymmetric)
% X: each column is a data point
% k: number of neighbors
% issymmetric: set W = (W+W')/2 if issymmetric=1
% W: similarity matrix

if nargin < 3
    issymmetric = 1;
end
if nargin < 2
    k = 5;
end

[~, n] = size(X);
D = L2_distance(X, X);
[DH, idx] = sort(D, 2);
% sort each row

W = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);
    di = D(i, id);
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end
if issymmetric == 1
    W = (W+W')/2;
end
W(isnan(W))=1;
B = sum(W,2);
W(B==0,:)= 1/n;
%normalize
A = diag(sum(W,2))\W;
end