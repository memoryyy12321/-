data1 = importdata('wine.txt');
data = data1(:,2:14);
[Z,MU,SIGMA] = zscore(data);%将读入的数据做标准化处理
[mappedX, mapping] = kernel_pca(Z,2,'gauss');
datanew = mappedX;
figure(1)%画出降维后的二维图
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
xlabel("第一主成分");
ylabel("第二主成分");
legend([spot_1,spot_2,spot_3],'Class 1','Class 2','Class 3');