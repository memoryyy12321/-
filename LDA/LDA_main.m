data1 = importdata('wine.txt');
data = data1(:,2:14);
[Z,MU,SIGMA] = zscore(data);%将读入的数据做标准化处理
class1 = Z(1:59,:);%第一类的训练集
class2 = Z(60:130,:);%第二类的训练集
class3 = Z(131:178,:);%第三类的训练集
%求期望
E_class1 = mean(class1);%第一类数据的期望矩阵
E_class2 = mean(class2);%第二类数据的期望矩阵
E_class3 = mean(class3);%第三类数据的期望矩阵
E_all = mean([E_class1;E_class2;E_class3]);%所有训练集的期望矩阵
%计算类间离散度矩阵
x1 = E_all-E_class1;
x2 = E_all-E_class2;
x3 = E_all-E_class3;
Sb = 59*x1'*x1/178+71*x2'*x2/178+48*x3'*x3/178;
%计算类内离散矩阵
y1 = 0;
for i = 1:59
    y1 = y1+(class1(i,:)-E_class1)'*(class1(i,:)-E_class1);
end
y2 = 0;
for i = 1:71
    y2 = y2+(class2(i,:)-E_class2)'*(class2(i,:)-E_class2);
end
y3 = 0;
for i = 1:48
    y3 = y3+(class3(i,:)-E_class3)'*(class3(i,:)-E_class3);
end
Sw = 59*y1/178+71*y2/178+48*y3/178;
%求最大特征值和特征向量
[V,L] = eig(inv(Sw)*Sb);
D = diag(L);%提取特征值矩阵的对角线元素，即特征值
rat1 = D./sum(D);%计算贡献率
rat2 = cumsum(D)./sum(D);%计算累计贡献率
figure(1)%绘制独立解释方差和累加解释方差
graph_1 = bar(rat1);%绘制独立解释方差
hold on
graph_2 = stairs(rat2);%绘制累加解释方差
xlabel("主成分索引");
ylabel("解释方差率");
legend([graph_1,graph_2],'独立解释方差','累加解释方差');
%第二个和第三个特征值最大，下面进行降维投影
tranMatrix = V(:,2:3);
datanew = Z*tranMatrix;%一定使用标准化后的特征矩阵，降维后标签不变
figure(2)
x11 = datanew(1:59,1);
y11 = datanew(1:59,2);
spot_1 = scatter(x11,y11,'k','d','filled');
hold on
x22 = datanew(60:130,1);
y22 = datanew(60:130,2);
spot_2 = scatter(x22,y22,'g','o','filled');
x33 = datanew(131:178,1);
y33 = datanew(131:178,2);
spot_3 = scatter(x33,y33,'b','^','filled');
xlabel("LD1");
ylabel("LD2");
legend([spot_1,spot_2,spot_3],'Class 1','Class 2','Class 3');