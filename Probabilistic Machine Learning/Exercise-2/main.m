clear all; close all; clc;

%% 

load('TempFieldDataSubset.mat');


%% Visualization of the test and train data

trainY = trainy;
testY = testy;

figure();
scatter3(trainX(1,:),trainX(2,:),trainY, 'r.');
hold on;
scatter3(testX(1,:),testX(2,:),testY, 'bo');
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
title('Visualizing the given dataset');
legend('Given Training Data','Given Test Data');
hold off;

%% Gaussian Processes with RBF Kernel
%derive covariance(k,k*,k**)
noise = 0.1;        %Assuming there's some noise in the data
sigma = 1;
l = 0.150;
trainX = trainX';
testX = testX';
%K = K(trainX,trainX)
K = zeros(size(trainX));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K(i,j) = sigma^2 * exp(-norm(trainX(i,:)-trainX(j,:))^2/(2*l^2));
    end
end
%K = K + noise*eye(length(trainX));

%k* = K(trainX,testX)
KStar = zeros(size(trainX,2),size(testX,2));
for i = 1:length(trainX)
    for j = 1:length(testX)
        KStar(i,j) = sigma^2 * exp(-norm(trainX(i,:)-testX(j,:))^2/(2*l^2));
    end
end

%k** = K(testX,testX)
KStarStar = zeros(size(testX,2));
for i = 1:length(testX)
    for j = 1:length(testX)
        KStarStar(i,j) = sigma^2 * exp(-norm(testX(i,:)-testX(j,:))^2/(2*l^2));
    end
end

mu = zeros(length(trainX),1);  %mean of train set
mu_Star = zeros(length(testX),1);  %mean of test set

% prediction

K_inverse= inv(K+noise^2*eye(size(K)));

E = mu_Star + transpose(KStar) * K_inverse * (trainY - mu);
var = KStarStar - transpose(KStar) * K_inverse * KStar;

% evaluation
MSE = norm(E-testY);
%max_abs_error = max(abs(E - testY));
fprintf('Error with RBF Kernel and mean = 0 is %d\n', MSE);

figure();
scatter3(testX(:,1),testX(:,2),testY,'b','.');
hold on;
scatter3(testX(:,1),testX(:,2),E,'r');
hold off;
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
legend('Original Temperature','Predicted Temperature')
title('RBF Kernel & mean = 0')


%% Gaussian Process with Periodic Kernel
sigma = 1;
l = 0.15;
p = 12;
%K = K(trainX,trainX)
K = zeros(length(trainX),length(trainX));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K(i,j) = sigma^2*exp((-2*sin(pi*norm(trainX(i,:)-trainX(j,:))/p)^2)/l^2);
    end
end
%K = K + noise*eye(length(trainX));

%k* = K(trainX,testX)
KStar = zeros(size(trainX,2),size(testX,2));
for i = 1:length(trainX)
    for j = 1:length(testX)
        KStar(i,j) = sigma^2*exp((-2*sin(pi*norm(trainX(i,:)-testX(j,:))/p)^2)/l^2);
    end
end


%k** = K(testX,testX)
KStarStar = zeros(size(testX,2));
for i = 1:length(testX)
    for j = 1:length(testX)
        KStarStar(i,j) = sigma^2*exp((-2*sin(pi*norm(testX(i,:)-testX(j,:))/p)^2)/l^2);
    end
end


K_inverse= inv(K+noise^2*eye(size(K)));

E = mu_Star + transpose(KStar) * K_inverse * (trainY - mu);
var = KStarStar - transpose(KStar) * K_inverse * KStar;

% evaluation
MSE = norm(E-testY);
%max_abs_error = max(abs(E - testY));
fprintf('Error with Periodic Kernel and mean = 0 is %d\n', MSE);

figure();
scatter3(testX(:,1),testX(:,2),testY,'b','.');
hold on;
scatter3(testX(:,1),testX(:,2),E,'r');
hold off;
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
legend('Original Temperature','Predicted Temperature')
title('Periodic Kernel and mean = 0')


%% Gaussian Process with Linear Kernel

sigma_b = 1;
c = [(max(trainX(:,1))+min(trainX(:,1)))/2, (max(trainX(:,2))+min(trainX(:,2)))/2];

%K = K(trainX,trainX)
K = zeros(length(trainX),length(trainX));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K(i,j) = sigma_b^2 + sigma^2*(trainX(i,:)-c)*(trainX(j,:)-c)';
    end
end
K = K + noise*eye(length(trainX));

%k* = K(trainX,testX)
KStar = zeros(size(trainX,2),size(testX,2));
for i = 1:length(trainX)
    for j = 1:length(testX)
        KStar(i,j) = sigma_b^2 + sigma^2*(trainX(i,:)-c)*(testX(j,:)-c)';
    end
end

%k** = K(testX,testX)
KStarStar = zeros(size(testX,2));
for i = 1:length(testX)
    for j = 1:length(testX)
        KStarStar(i,j) = sigma_b^2 + sigma^2*(testX(i,:)-c)*(testX(j,:)-c)';
    end
end

K_inverse= inv(K+noise^2*eye(size(K)));

E = mu_Star + transpose(KStar) * K_inverse * (trainY - mu);
var = KStarStar - transpose(KStar) * K_inverse * KStar;

% evaluation
MSE = norm(E-testY);
%max_abs_error = max(abs(E - testY));
fprintf('Error with Linear Kernel and mean = 0 is %d\n', MSE);

figure();
scatter3(testX(:,1),testX(:,2),testY,'b','.');
hold on;
scatter3(testX(:,1),testX(:,2),E,'r');
hold off;
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
legend('Original Temperature','Predicted Temperature')
title('Linear Kernel and mean = 0')



%% Introducing different different Means functions
%derive covariance
noise = 0.1;
sigma = 1;
l = 0.150;

%K = K(trainX,trainX)
K = zeros(size(trainX,2));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K(i,j) = sigma^2 * exp(-norm(trainX(i,:)-trainX(j,:))^2/(2*l^2));
    end
end
K = K + noise*eye(length(trainX));

%k* = K(trainX,testX)
KStar = zeros(size(trainX,2),size(testX,2));
for i = 1:length(trainX)
    for j = 1:length(testX)
        KStar(i,j) = sigma^2 * exp(-norm(trainX(i,:)-testX(j,:))^2/(2*l^2));
    end
end

%k** = K(testX,testX)
KStarStar = zeros(size(testX,2));
for i = 1:length(testX)
    for j = 1:length(testX)
        KStarStar(i,j) = sigma^2 * exp(-norm(testX(i,:)-testX(j,:))^2/(2*l^2));
    end
end

mu = mean(trainY).*ones(length(trainX),1);
mu_Star = mean(trainY).*ones(length(testX),1);

% prediction
K_inverse= inv(K+noise^2*eye(size(K)));

E = mu_Star + transpose(KStar) * K_inverse * (trainY - mu);
var = KStarStar - transpose(KStar) * K_inverse * KStar;

% evaluation
MSE = norm(E-testY);
fprintf('Error with RBF Kernel and mean = mean(y) is %d\n', MSE);

figure();
scatter3(testX(:,1),testX(:,2),testY,'b','.');
hold on;
scatter3(testX(:,1),testX(:,2),E,'r');
hold off;
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
legend('Original Temperature','Predicted Temperature')
title('RBF Kernel and mean = mean(y)')


%% ridge regression based linear mean function
% Normalizing -> using mapstd command as instructed
[trainX_Norm,trainX_settings] = mapstd(trainX);
[testX_Norm,testX_settings] = mapstd(testX);

% implemeting bias or offset term
%trainX_bias = [ones(1,size(trainX_Norm,2));trainX_Norm];
trainX_bias = [ones(length(trainX),1), trainX_Norm];
%testX_bias = [ones(1,size(testX_Norm,2)); testX_Norm];
testX_bias = [ones(length(testX),1), testX_Norm];

% Implementing Ridge Regression-> weight vector, w = (A'A + lambda*I)^(-1)/A'y
l = 1;  %let lambda = l
A = (trainX_bias);
w = (A'*A + l*eye(size(A',1)))\(A'*trainY);

% calculation of mu and mu_star
mu = (trainX_bias*w);
mu_Star = (testX_bias*w);

K_inverse= inv(K+noise^2*eye(size(K))); %K is already calculated in the above RBF Kernel

E = mu_Star + transpose(KStar) * K_inverse * (trainY - mu);
var = KStarStar - transpose(KStar) * K_inverse * KStar;

% evaluation
MSE = norm(E-testY);
%max_abs_error = max(abs(E - testY));
fprintf('Error with RBF Kernel and mean from Ridge regression is %d\n', MSE);

figure();
scatter3(testX(:,1),testX(:,2),testY,'b','.');
hold on;
scatter3(testX(:,1),testX(:,2),E,'r');
hold off;
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
legend('Original Temperature','Predicted Temperature')
title('RBF GP with mean from Linear kernel or ridge regression')


%% Meta learning
params = [1,0.72];
parameters_initial = [1,0.15];
cost_function = @(params) LogLikelihood(trainX,trainY,params);
options = optimset('Display','iter', 'GradObj', 'on','MaxIter',30);
lb = [0;0];
ub = [1;100];
%[parameters_opt,cost_opt] = fmincon(cost_function,parameters_initial,[],[],[],[],lb,ub,[],options);
[parameters_opt,cost_opt,exitflag] = fminunc(cost_function,parameters_initial,options);

sigma = parameters_opt(1);
l = parameters_opt(2);
%K = K(trainX,trainX)
K = zeros(size(trainX,2));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K(i,j) = sigma^2 * exp(-norm(trainX(i,:)-trainX(j,:))^2/(2*l^2));
    end
end
K = K + noise*eye(length(trainX));

%k* = K(trainX,testX)
KStar = zeros(size(trainX,2),size(testX,2));
for i = 1:length(trainX)
    for j = 1:length(testX)
        KStar(i,j) = sigma^2 * exp(-norm(trainX(i,:)-testX(j,:))^2/(2*l^2));
    end
end

%k** = K(testX,testX)
KStarStar = zeros(size(testX,2));
for i = 1:length(testX)
    for j = 1:length(testX)
        KStarStar(i,j) = sigma^2 * exp(-norm(testX(i,:)-testX(j,:))^2/(2*l^2));
    end
end

% calculation of mu and mu_star
mu = (trainX_bias*w);
mu_Star = (testX_bias*w);

%prediction
K_inverse= inv(K+noise^2*eye(size(K)));
E = mu_Star + transpose(KStar) * K_inverse * (trainY - mu);
var = KStarStar - transpose(KStar) * K_inverse * KStar;

% evaluation
MSE = norm(E-testY);
%fprintf('Error with RBF Kernel with Meta Learning is %d\n', MSE);

figure();
scatter3(testX(:,1),testX(:,2),testY,'b','.');
hold on;
scatter3(testX(:,1),testX(:,2),E,'r');
hold off;
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
legend('Original Temperature','Predicted Temperature');
title('Loglikelihood optimized RBF kernel and mean as output of Linear Regression');







