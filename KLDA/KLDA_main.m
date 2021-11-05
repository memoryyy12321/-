data1 = importdata('wine.txt');
data = data1(:,2:14);
[Z,MU,SIGMA] = zscore(data);%���������������׼������
label = data1(:,1);
datanew = gda(Z,label,2,'gauss');
figure(1)%������ά��Ķ�άͼ
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
xlabel('��һ���ɷ�');
ylabel('�ڶ����ɷ�');
legend([spot_1,spot_2,spot_3],'Class 1','Class 2','Class 3');