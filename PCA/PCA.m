function [Y,eigVector,eigValue] = PCA(X,d)
%%主成分分析
Sx = cov(X);
[V,D] = eig(Sx);
eigValue = diag(D);
[eigValue,IX] = sort(eigValue,'descend');
eigVector = V(:,IX);
%%标准化
norm_eigVector = sqrt(sum(eigVector.^2));
eigVector = eigVector./repmat(norm_eigVector,size(eigVector,1),1);
%%PCA降维
eigVector = eigVector(:,1:d);
Y = X*eigVector;
