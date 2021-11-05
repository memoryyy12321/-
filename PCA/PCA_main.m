data1 = importdata('wine.txt');
data = data1(:,2:14);
%�ڽ���pca��άǰ��ǰ��������ֵAlcohol(�ƾ���)��Malic acid(ƻ����)��Ash(��)Ϊ������۲���ռ�ֲ�
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
xlabel('Alcohol(�ƾ���)');
ylabel('Malic acid(ƻ����)');
zlabel('Ash(��)');
legend([spot_1,spot_2,spot_3],'Class 1','Class 2','Class 3');
[Z,MU,SIGMA] = zscore(data);%���������������׼������
covMat = cov(Z);%�������ϵ������
[COEFF,latent,explained] = pcacov(covMat);%����pcacov���������ɷַ���
cumsum(explained);
figure(2)%���ƶ������ͷ�����ۼӽ��ͷ���
graph_1 = bar(explained);%���ƶ������ͷ���
hold on
graph_2 = stairs(cumsum(explained));%�����ۼӽ��ͷ���
xlabel('���ɷ�����');
ylabel('���ͷ�����');
legend([graph_1,graph_2],'�������ͷ���','�ۼӽ��ͷ���');
%�����ά���������������������ά���ɢ��ͼ
%------����ʹ���Զ��庯��------
%datanew = PCA(Z,2);
%����ʹ����д�ĺ���
tranMatrix = COEFF(:,1:2);
datanew = Z*tranMatrix;%һ��ʹ�ñ�׼������������󣬽�ά���ǩ����
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
xlabel('��һ���ɷ�');
ylabel('�ڶ����ɷ�');
legend([spot_4,spot_5,spot_6],'Class 1','Class 2','Class 3');