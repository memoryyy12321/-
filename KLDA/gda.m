function mappedX = gda(X, Y, no_dims, varargin)
%% --------------------------程序说明-------------------------------
%此程序用以实现广义判别分析（Generalized Discriminant Analysis，GDA）
%格式：
%	mappedX = gda(X, Y, no_dims)
%	mappedX = gda(X, Y, no_dims, kernel)
%	mappedX = gda(X, Y, no_dims, kernel, param1)
%	mappedX = gda(X, Y, no_dims, kernel, param1, param2)
%X是要执行GDA的数据
%Y是X相应的标签
%kernel的值决定使用的核函数，可取为 'linear', 'gauss', 'poly', (默认 = 'gauss')
%% ---------------------------正式程序------------------------------
%---------------------------------检查输入--------------------------------- 
if ~exist('no_dims', 'var')
    no_dims = 2;
end
kernel = 'gauss';
param1 = 1;
param2 = 0;
if length(varargin) > 0 & strcmp(class(varargin{1}), 'char'), kernel = varargin{1}; end 
if length(varargin) > 1 & strcmp(class(varargin{2}), 'double'), param1 = varargin{2}; end
if length(varargin) > 2 & strcmp(class(varargin{3}), 'double'), param2 = varargin{3}; end
%--------------------------------根据标签对数据分类-------------------------
[foo, bar, Y] = unique(Y, 'rows');
[n, dim] = size(X);
nclass = max(Y);
[foo, ind] = sort(Y);
Y = Y(ind);
X = X(ind,:);
%---------------------------------计算kernel矩阵---------------------------
disp('计算kernel矩阵...');
K = gram(X, X, kernel, param1, param2);
%--------------------------------计算中心矩阵------------------------------ 
ell = size(X, 1);
D = sum(K) / ell;
E = sum(D) / ell;
J = ones(ell, 1) * D;
K = K - J - J' + E * ones(ell, ell);
%---------------------------kernel矩阵特征向量分解--------------------------
disp('kernel矩阵特征向量分解...');
K(isnan(K)) = 0;
K(isinf(K)) = 0;
[P, gamma] = eig(K);
if size(P, 2) < n
    error('kernel矩阵为奇异矩阵，无解');
end
%---------------------------按降序排列特征值和向量--------------------------
[gamma, ind] = sort(diag(gamma), 'descend');
P = P(:,ind);
%----------------------移除具有相对较小特征值的特征向量----------------------
minEigv = max(gamma) / 1e5;
ind = find(gamma > minEigv);
P = P(:,ind);
gamma = gamma(ind);
rankK = length(ind);
%----------------------------重新计算核矩阵---------------------------------
K = P * diag(gamma) * P';
%-----------------------------构造对角矩阵W--------------------------------
W = [];
for i=1:nclass
    num_data_class = length(find(Y == i));
    W = blkdiag(W, ones(num_data_class) / num_data_class);
end  
%-----------------------确定数据的目标维度----------------------------------
old_no_dims = no_dims;
no_dims = min([no_dims rankK nclass]);
if old_no_dims > no_dims
    warning(['目标维数降为 ' num2str(no_dims) '.']);
end
%-----------------------------矩阵的特征分解--------------------------------
disp('执行GDA特征分析...');
[Beta, lambda] = eig(P' * W * P);
lambda = diag(lambda);
%------------------------按降序排列特征值和特征向量-------------------------
[lambda, ind] = sort(lambda, 'descend');
Beta = Beta(:,ind(1:no_dims));
%---------------------------计算最终嵌入的特征空间--------------------------
mappedX = P * diag(1 ./ gamma) * Beta;
for i=1:no_dims
    mappedX(:,i) = mappedX(:,i) / sqrt(mappedX(:,i)' * K * mappedX(:,i));
end
end