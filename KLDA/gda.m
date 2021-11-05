function mappedX = gda(X, Y, no_dims, varargin)
%% --------------------------����˵��-------------------------------
%�˳�������ʵ�ֹ����б������Generalized Discriminant Analysis��GDA��
%��ʽ��
%	mappedX = gda(X, Y, no_dims)
%	mappedX = gda(X, Y, no_dims, kernel)
%	mappedX = gda(X, Y, no_dims, kernel, param1)
%	mappedX = gda(X, Y, no_dims, kernel, param1, param2)
%X��Ҫִ��GDA������
%Y��X��Ӧ�ı�ǩ
%kernel��ֵ����ʹ�õĺ˺�������ȡΪ 'linear', 'gauss', 'poly', (Ĭ�� = 'gauss')
%% ---------------------------��ʽ����------------------------------
%---------------------------------�������--------------------------------- 
if ~exist('no_dims', 'var')
    no_dims = 2;
end
kernel = 'gauss';
param1 = 1;
param2 = 0;
if length(varargin) > 0 & strcmp(class(varargin{1}), 'char'), kernel = varargin{1}; end 
if length(varargin) > 1 & strcmp(class(varargin{2}), 'double'), param1 = varargin{2}; end
if length(varargin) > 2 & strcmp(class(varargin{3}), 'double'), param2 = varargin{3}; end
%--------------------------------���ݱ�ǩ�����ݷ���-------------------------
[foo, bar, Y] = unique(Y, 'rows');
[n, dim] = size(X);
nclass = max(Y);
[foo, ind] = sort(Y);
Y = Y(ind);
X = X(ind,:);
%---------------------------------����kernel����---------------------------
disp('����kernel����...');
K = gram(X, X, kernel, param1, param2);
%--------------------------------�������ľ���------------------------------ 
ell = size(X, 1);
D = sum(K) / ell;
E = sum(D) / ell;
J = ones(ell, 1) * D;
K = K - J - J' + E * ones(ell, ell);
%---------------------------kernel�������������ֽ�--------------------------
disp('kernel�������������ֽ�...');
K(isnan(K)) = 0;
K(isinf(K)) = 0;
[P, gamma] = eig(K);
if size(P, 2) < n
    error('kernel����Ϊ��������޽�');
end
%---------------------------��������������ֵ������--------------------------
[gamma, ind] = sort(diag(gamma), 'descend');
P = P(:,ind);
%----------------------�Ƴ�������Խ�С����ֵ����������----------------------
minEigv = max(gamma) / 1e5;
ind = find(gamma > minEigv);
P = P(:,ind);
gamma = gamma(ind);
rankK = length(ind);
%----------------------------���¼���˾���---------------------------------
K = P * diag(gamma) * P';
%-----------------------------����ԽǾ���W--------------------------------
W = [];
for i=1:nclass
    num_data_class = length(find(Y == i));
    W = blkdiag(W, ones(num_data_class) / num_data_class);
end  
%-----------------------ȷ�����ݵ�Ŀ��ά��----------------------------------
old_no_dims = no_dims;
no_dims = min([no_dims rankK nclass]);
if old_no_dims > no_dims
    warning(['Ŀ��ά����Ϊ ' num2str(no_dims) '.']);
end
%-----------------------------����������ֽ�--------------------------------
disp('ִ��GDA��������...');
[Beta, lambda] = eig(P' * W * P);
lambda = diag(lambda);
%------------------------��������������ֵ����������-------------------------
[lambda, ind] = sort(lambda, 'descend');
Beta = Beta(:,ind(1:no_dims));
%---------------------------��������Ƕ��������ռ�--------------------------
mappedX = P * diag(1 ./ gamma) * Beta;
for i=1:no_dims
    mappedX(:,i) = mappedX(:,i) / sqrt(mappedX(:,i)' * K * mappedX(:,i));
end
end