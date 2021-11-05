function y = kernel_function(v, X, center, kernel, param1, param2, type)
%% --------------------------程序说明-------------------------------
%该程序用来计算(K * X)的和，其中X是特征数据
%格式：
%   y = kernel_function(v, X, center, kernel, param1, param2)
%该函数计算（K*v）元素的和，其中v是K的可能特征向量
%此函数用来确保基于Kernel的PCA中EIGS能够被使用
%参数X为数据集
%kernel为核函数的名称（默认值为“gauss”）
%param1和param2指的是kernel具体的参数
%% --------------------------具体程序-------------------------------
%-------------------------------判断输入-----------------------------------
if ~exist('center', 'var')
    center = 0;
end
if ~exist('type', 'var')
    type = 'Normal';
end
if ~strcmp(type, 'ColumnSums'), fprintf('.'); end    
%----------------------------如果没有明确核函数-----------------------------
if nargin == 2 || strcmp(kernel, 'none')
    kernel = 'linear';
end
%--------------------------------构造结果向量-------------------------------
y = zeros(1, size(X, 1));
n = size(X, 2);
%----------------------------根据kernel选择计算方式-------------------------
switch kernel
%% --------------------------线性核函数-----------------------------
    case 'linear'
        %----------------------检索K的居中信息------------------------------
        if center || strcmp(type, 'ColumnSums')
            column_sum = zeros(1, n);
            for i=1:n
                %--------------计算kernel矩阵的单行-------------------------
                K = X(:,i)' * X;
                column_sum = column_sum + K;
            end
            %------------------计算整个内核的中心常数-----------------------
            total_sum = ((1 / n^2) * sum(column_sum));
        end
        if ~strcmp(type, 'ColumnSums')
            %-----------------------计算乘积K*v----------------------------
            for i=1:n
                %-----------------计算核矩阵的单行--------------------------
                K = X(:,i)' * X;
                %-----------------kernel矩阵的中心行------------------------
                if center
                    K = K - ((1 / n) .* column_sum) - ((1 / n) .* column_sum(i)) + total_sum;
                end
                %----------------------计算乘积和--------------------------
                y(i) = K * v;
            end
        else
            %--------------------------返回列和----------------------------
            y = column_sum;
        end
%% ---------------------------多项式核函数---------------------------
    case 'poly'
        %---------------------------初始化一些变量--------------------------
        if ~exist('param1', 'var'), param1 = 1; param2 = 3; end            
        %---------------------------检索K的居中信息------------------------- 
        if center || strcmp(type, 'ColumnSums')
            column_sum = zeros(1, n);
            for i=1:n
                %------------------计算核矩阵的列和-------------------------
                K = X(:,i)' * X;
                K = (K + param1) .^ param2;
                column_sum = column_sum + K;
            end
            %---------------------计算整个内核的中心常数--------------------
            total_sum = ((1 / n^2) * sum(column_sum));
        end       
        if ~strcmp(type, 'ColumnSums')
            %-----------------------计算乘积K*v----------------------------
            for i=1:n
                % Compute row of the kernel matrix
                K = X(:,i)' * X;
                K = (K + param1) .^ param2;

                %-----------------计算kernel矩阵的行-----------------------
                if center
                    K = K - ((1 / n) .* column_sum) - ((1 / n) .* column_sum(i)) + total_sum;
                end
                %--------------------计算乘积之和--------------------------
                y(i) = K * v;
            end
        else
            %-------------------------返回列和-----------------------------
            y = column_sum;
        end
%% ---------------------------高斯核函数----------------------------
    case 'gauss'
        %-------------------------初始化一些变量---------------------------
        if ~exist('param1', 'var'), param1 = 1; end
        % Retrieve information for centering of K
        if center || strcmp(type, 'ColumnSums')
            column_sum = zeros(1, n);
            for i=1:n
                %--------------------检索K的居中信息-----------------------
                K = L2_distance(X(:,i), X);
                K = exp(-(K.^2 / (2 * param1.^2)));
                column_sum = column_sum + K;
            end
            %----------------------计算整个内核的中心常数-------------------
            total_sum = ((1 / n^2) * sum(column_sum));
        end
        if ~strcmp(type, 'ColumnSums')
            %------------------------计算乘积K*v---------------------------
            for i=1:n
                %------------------计算核矩阵的单行------------------------
                K = L2_distance(X(:,i), X);
                K = exp(-(K.^2 / (2 * param1.^2)));
                %-----------------核矩阵的中心行---------------------------                 
                if center
                    K = K - ((1 / n) .* column_sum) - ((1 / n) .* column_sum(i)) + total_sum;
                end
                %----------------------计算乘积和--------------------------
                y(i) = K * v;
            end
        else
            %---------------------------返回列和---------------------------
            y = column_sum;
        end
    otherwise
        error('未知核函数.');
end
end