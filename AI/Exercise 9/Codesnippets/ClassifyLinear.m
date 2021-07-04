%Wine quality data set.
%--------------------------------------------------------------------------
%Distinguish between good wines and not good wines. 
%A wine is good if its score is 7 or higher.

%read data
load('Data.mat');

%TODO: Define X and y for the learning model


%TODO: Implement 5-fold Cross Validation. Iterate over the folds to split 
%the data X and y into X_train, y_train, X_test and y_test. 


%Example of linear SVM training (fitcsvm) and prediction with trained
%model (predict). We use a radial basis function (rbf) kernel here.

%training
SVMModel_linear=fitcsvm(X_train,y_train);

%test
y_pred=predict(SVMModel_linear,X_test);
















































