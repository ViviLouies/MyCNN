% B=flipdim(A,dim)
% A��ʾһ������dimָ����ת��ʽ��
% dimΪ1����ʾÿһ�н����������У�
% dimΪ2����ʾÿһ�н����������С�
function X = rot180(X)
X = flipdim(flipdim(X, 1), 2);
end