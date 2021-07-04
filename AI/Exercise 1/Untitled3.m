clc;clear all

f = [-30 -25];
A = [1 1;5 2;1 0; 0 1; -1 0; 0 -1];
b = [10 30 6 9 0 0];

%Aeq = [];
%beq = [];

%lb = [0 0];
%ub = [];

res = linprog(f,A,b);