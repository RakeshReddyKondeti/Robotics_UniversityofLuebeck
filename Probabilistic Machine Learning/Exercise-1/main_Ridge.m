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

%% Implementing Ridge Regression-> weight vector, w = (A'A + lambda*I)^(-1)/A'y
l = 1;  %let lambda = l
A = (trainX_bias)';
w = (A'*A + l*eye(size(A',1)))\(A'*trainy);


%% calculation and visualization of predicted temperature
pred_y = (testX_bias'*w);

figure();
plot3(testX(1,:),testX(2,:),testy, 'r.');
hold on;
plot3(testX(1,:),testX(2,:),pred_y,'g.');
xlabel('Latitute');
ylabel('Logitude');
zlabel('Temperature');
title('Temperature -  Ridge Linear Regression')
legend('Train dataset','seperating boundary using ridge regression');
hold off;


%% Training and Test errors with increasing training data size
train_error = zeros(size(trainX,2),1);
test_error = zeros(size(testX,2),1);

for i = 1:size(trainX_bias,2)
    A = trainX_bias(:,1:i)';
    y = trainy(1:i);
    w = (A'*A + l*eye(size(A',1)))\(A'*y);
    pred_testy = (testX_bias'*w); %prediction of the test values
    test_error(i) = mean((pred_testy-testy).^2); %MSE of test sample
    pred_trainy = (A*w);%prediction of the training values
    train_error(i) = mean((pred_trainy-y).^2); %MSE of the train sample
end

%ploting the errors
figure();
plot(1:length(test_error),test_error(1:length(test_error)), 'k');
hold on;
plot((1:length(train_error)),train_error(1:length(train_error)),'m');
xlabel('Number of Training Points');
ylabel('Error');
hold off;
legend('Test error','Training error');
title(' Error from Ridge Regression Model');
axis([0 9000 0 100]);