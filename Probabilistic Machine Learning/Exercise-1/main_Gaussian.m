clear all;
close all;
clc;


%% loading the dataset
load('TempFieldDataSubset.mat')

%% Visualizating the test and train data

figure();
plot3(trainX(1,:),trainX(2,:),trainy, 'r.');
hold on;
plot3(testX(1,:),testX(2,:),testy, 'b.');
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
title('Visualizing the given dataset');
legend('Given Training Data','Given Test Data');
hold off;


%% Normalizing -> using mapstd command as instructed
[trainX_Norm,trainX_settings] = mapstd(trainX);
[testX_Norm,testX_settings] = mapstd(testX);


%% implemeting bias or offset term
trainX_bias = [ones(1,size(trainX_Norm,2));trainX_Norm];
testX_bias = [ones(1,size(testX_Norm,2));testX_Norm];

%% Multivariate regression

basis = [2,5,10,15,20];
err_train = [];
err_test = [];

for no_of_basis = basis
    fprintf('%f',no_of_basis);
    
    mu_x1 = linspace(min(trainX_Norm(1,:)),max(trainX_Norm(1,:)),no_of_basis);
    mu_x2 = linspace(min(trainX_Norm(2,:)),max(trainX_Norm(2,:)),no_of_basis);
   
    inv_cov = inv(cov(trainX_Norm(1,:),trainX_Norm(2,:)));
    
    
    for i = 1:length(mu_x1)
        for j = 1:length(mu_x2)
           mu = [mu_x1(i);mu_x2(j)];
           for k = 1:length(trainX)
               phi_train(k,1) = exp(-0.5*(trainX_Norm(:,k)-mu)'*inv_cov*(trainX_Norm(:,k)-mu));       
           end
           phi_train = [ones(size(trainX,2),1) phi_train];
           for k = 1:length(testX)
                phi_test(k,1) = exp(-0.5*(testX_Norm(:,k)-mu)'*inv_cov*(testX_Norm(:,k)-mu));        
           end
           phi_test = [ones(size(testX,2),1) phi_test];
        end
    end
    
    lambda = 1;
    w = ((phi_train)'*phi_train + lambda*eye(size(phi_train',1)))\(phi_train'*trainy);
    
    pred_testy = phi_test*w;
    pred_trainy = phi_train*w;
    
    err_train = [err_train mean((pred_trainy-trainy).^2)];
    err_test = [err_test mean((pred_testy-testy).^2)];
end


figure();
plot(basis, err_test, 'b');
hold on;
plot(basis, err_train,'r');
xlabel('Number of Basis');
ylabel('Error');
title('Error with respect to the number of Basis');
hold off;
legend('Test error','Training error');


