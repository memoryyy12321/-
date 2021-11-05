data1 = importdata('wine.txt');
data = data1(:,2:14);
%在进行pca降维前以前三个特征值Alcohol(酒精度)、Malic acid(苹果酸)、Ash(灰)为坐标轴观察其空间分布
figure(1)
x1 = data(1:59,1);
y1 = data(1:59,2);
z1 = data(1:59,3);
spot_1 = scatter3(x1,y1,z1,'k','d','filled');
hold on
x2 = data(60:130,1);
y2 = data(60:130,2);
z2 = data(60:130,3);
spot_2 = scatter3(x2,y2,z2,'g','o','filled');
hold on
x3 = data(131:178,1);
y3 = data(131:178,2);
z3 = data(131:178,3);
spot_3 = scatter3(x3,y3,z3,'b','^','filled');
xlabel('Alcohol(酒精度)');
ylabel('Malic acid(苹果酸)');
zlabel('Ash(灰)');
legend([spot_1,spot_2,spot_3],'Class 1','Class 2','Class 3');
[Z,MU,SIGMA] = zscore(data);%将读入的数据做标准化处理
covMat = cov(Z);%计算相关系数矩阵
[COEFF,latent,explained] = pcacov(covMat);%调用pcacov函数作主成分分析
cumsum(explained);
figure(2)%绘制独立解释方差和累加解释方差
graph_1 = bar(explained);%绘制独立解释方差
hold on
graph_2 = stairs(cumsum(explained));%绘制累加解释方差
xlabel('主成分索引');
ylabel('解释方差率');
legend([graph_1,graph_2],'独立解释方差','累加解释方差');
%求出降维后的特征向量，并画出降维后的散点图
%------下面使用自定义函数------
%datanew = PCA(Z,2);
%下面使用手写的函数
tranMatrix = COEFF(:,1:2);
datanew = Z*tranMatrix;%一定使用标准化后的特征矩阵，降维后标签不变
figure(3)
x11 = datanew(1:59,1);
y11 = datanew(1:59,2);
spot_4 = scatter(x11,y11,'k','d','filled');
hold on
x22 = datanew(60:130,1);
y22 = datanew(60:130,2);
spot_5 = scatter(x22,y22,'g','o','filled');
x33 = datanew(131:178,1);
y33 = datanew(131:178,2);
spot_6 = scatter(x33,y33,'b','^','filled');
xlabel('第一主成分');
ylabel('第二主成分');
legend([spot_4,spot_5,spot_6],'Class 1','Class 2','Class 3');