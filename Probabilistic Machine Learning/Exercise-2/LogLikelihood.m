function [cost, grad] = loglikelihood(trainX,trainY,params)
     
sigma = params(1);
l = params(2);
noise = 0.1;

%K = K(trainX,trainX)
K = zeros(size(trainX,2));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K(i,j) = sigma^2 * exp(-norm(trainX(i,:)-trainX(j,:))^2/(2*l^2));
    end
end
K = K + noise*eye(length(trainX));

n = length(trainX);
K_inv = inv(K);

cost = 0.5*(trainY'/K)*trainY + 0.5*log(norm(K)) + (n/2)*log(2*pi);

alpha = K\trainY;
d = 0.001;      %step-size
sigma_new = sigma + d;
l_new = l + d;

%K_new_sigma = K(trainX,trainX)
K_new_sigma = zeros(size(trainX,2));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K_new_sigma(i,j) = sigma_new^2 * exp(-norm(trainX(i,:)-trainX(j,:))^2/(2*l^2));
    end
end

grad = zeros(length(params),1);
grad(1) = 0.5*trace( (alpha*alpha'-K_inv)*((K_new_sigma-K)./d) );

%K = K(trainX,trainX)
K_new_l = zeros(size(trainX,2));
for i = 1:length(trainX)
    for j = 1:length(trainX)
        K_new_l(i,j) = sigma^2 * exp(-norm(trainX(i,:)-trainX(j,:))^2/(2*l_new^2));
    end
end

grad(2) = 0.5*trace( (alpha*alpha'-K_inv)*((K_new_l-K)./d) );

    
end