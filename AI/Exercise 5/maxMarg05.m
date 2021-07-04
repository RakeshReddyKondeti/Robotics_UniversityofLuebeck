function [exitflag, w, d, margin, dists, alphas, sv] = maxMarg05( X, y )
% Input
% -----
%
% X        ... Data points and class labels.
%              [ x_11, x_12;
%                x_21, x_22;
%                x_31, x_32;
%                ...              ]
%
% y        ... Class labels.
%              [ s_1; s_2; s_3; ... ]

% Output
% ------
%
% exitflag ... Exitflag of quadprog.
%
% w        ... Weight vector.
%
% d        ... Bias of Separating Plane.
%
% margin   ... Margin.
%
% dists    ... Distances of each data point to the separating plane.
%
% alphas   ... Lagrange multipliers.
%
% sv       ... Indices of support vectors.

% 1.    Fabian Domberg 
% 2.	Rakesh Reddy
% 3.	Tim-Henrik Traving
% 4.	Harsh Yadav

% YOUR IMPLEMENTATION GOES HERE...

n = size(X,1);
H_d = zeros(n,n);
for i = 1:n
    for j = 1:n
        H_d(i,j) = (y(i)*y(j)).*(X(i,:)*X(j,:)');
    end
end
f_d = -1.*ones(n,1);
A_d= zeros(n,n);
b_d = zeros(n,1);
Aeq_d = y;
beq_d = 0;
lb = zeros(n,1);

[alphas,~,exitflag] = quadprog(H_d,f_d,A_d,b_d,Aeq_d,beq_d,lb);
sv = X((abs(alphas)>exp(-10)),:);
w = (y.*alphas')*X;
margin = 1/norm(w);
wx = w*X';
d = -0.5*(max(wx(y ==-1))+min(wx(y==1)));
w = margin.*w;
d = margin*d;
dists = (X*w' + ones(n,1).*d);

end