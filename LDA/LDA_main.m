data1 = importdata('wine.txt');
data = data1(:,2:14);
[Z,MU,SIGMA] = zscore(data);%���������������׼������
class1 = Z(1:59,:);%��һ���ѵ����
class2 = Z(60:130,:);%�ڶ����ѵ����
class3 = Z(131:178,:);%�������ѵ����
%������
E_class1 = mean(class1);%��һ�����ݵ���������
E_class2 = mean(class2);%�ڶ������ݵ���������
E_class3 = mean(class3);%���������ݵ���������
E_all = mean([E_class1;E_class2;E_class3]);%����ѵ��������������
%���������ɢ�Ⱦ���
x1 = E_all-E_class1;
x2 = E_all-E_class2;
x3 = E_all-E_class3;
Sb = 59*x1'*x1/178+71*x2'*x2/178+48*x3'*x3/178;
%����������ɢ����
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
%���������ֵ����������
[V,L] = eig(inv(Sw)*Sb);
D = diag(L);%��ȡ����ֵ����ĶԽ���Ԫ�أ�������ֵ
rat1 = D./sum(D);%���㹱����
rat2 = cumsum(D)./sum(D);%�����ۼƹ�����
figure(1)%���ƶ������ͷ�����ۼӽ��ͷ���
graph_1 = bar(rat1);%���ƶ������ͷ���
hold on
graph_2 = stairs(rat2);%�����ۼӽ��ͷ���
xlabel("���ɷ�����");
ylabel("���ͷ�����");
legend([graph_1,graph_2],'�������ͷ���','�ۼӽ��ͷ���');
%�ڶ����͵���������ֵ���������н�άͶӰ
tranMatrix = V(:,2:3);
datanew = Z*tranMatrix;%һ��ʹ�ñ�׼������������󣬽�ά���ǩ����
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