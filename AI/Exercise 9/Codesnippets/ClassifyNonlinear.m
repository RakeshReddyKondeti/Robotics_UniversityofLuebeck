%Wine quality data set.
%--------------------------------------------------------------------------
%Distinguish between good wines and not good wines. 
%A wine is good if its score is 7 or higher.

%read data
load('Data.mat');

%TODO: Define X and y for the learning model


%TODO: Implement 5-fold Cross Validation. Iterate over the folds to split 
%the data X and y into X_train, y_train, X_test and y_test. 


%Example of nonlinear SVM training (fitcsvm) and prediction with trained
%model (predict). We use a radial basis function (rbf) kernel here.

C=1;
gamma=10;

%training
SVMModel_nonlinear=fitcsvm(X_train,y_train,'BoxConstraint',C,'KernelFunction','rbf','KernelScale',gamma);

%test
y_pred=predict(SVMModel_nonlinear,X_test);















































