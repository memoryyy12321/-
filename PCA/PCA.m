function [Y,eigVector,eigValue] = PCA(X,d)
%%���ɷַ���
Sx = cov(X);
[V,D] = eig(Sx);
eigValue = diag(D);
[eigValue,IX] = sort(eigValue,'descend');
eigVector = V(:,IX);
%%��׼��
norm_eigVector = sqrt(sum(eigVector.^2));
eigVector = eigVector./repmat(norm_eigVector,size(eigVector,1),1);
%%PCA��ά
eigVector = eigVector(:,1:d);
Y = X*eigVector;
