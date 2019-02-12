%% Initializations
% load('household_power_consumption_.csv');
clear all;
data = load('household_v3.txt');
X = data(:,1:6)';
y = data(:,9);
N = length(y);
d = size(X,1);
D = 0;

Z = X * diag(y);
lambda = 0.1;
num_iters = 1000;
epsilon = 0.1;
L = 1/(4*N) * norm(Z,'fro')^2 + 2*lambda;
alpha = 0.1;
w = zeros(d,size(lambda,2));
 
 %% GD
 tic
 for i = 1:size(lambda,2)
    wGD(:,i) = gradientDescentMulti(Z, w(:,i), alpha(i), num_iters, lambda(i), epsilon)
    f(i) = 1/N * log(1 + exp(-wGD(:,i)' * Z)) * ones(N,1) + lambda(i) * (norm(wGD(:,i))^2 - D);
    fprintf("f = %f \n", f);
 end
 toc
 %% SGD
 tic
for i = 1:size(lambda,2)
    wSG(:,i) = stochasticGradient(Z, w(:,i), alpha(i), num_iters, lambda(i), epsilon)
    f(i) = 1/N * log(1 + exp(-wSG(:,i)' * Z)) * ones(N,1) + lambda(i) * (norm(wSG(:,i))^2 - D);
    fprintf("f = %f \n", f);
end
 toc
 %% SAG
 tic
for i = 1:size(lambda,2)
    wSA(:,i) = stochasticAverageGradient(Z, w(:,i), alpha(i), num_iters, lambda(i), epsilon)
    f(i) = 1/N * log(1 + exp(-wSA(:,i)' * Z)) * ones(N,1) + lambda(i) * (norm(wSA(:,i))^2 - D);
    fprintf("f = %f \n", f);
end 
 toc
 %% SVRG
tic

for i = 1:size(lambda,2)
    wSVR(:,i) = stochasticVarReduced(Z, w(:,i), alpha(i), num_iters, lambda(i), epsilon)
    f(i) = 1/N * log(1 + exp(-wSVR(:,i)' * Z)) * ones(N,1) + lambda(i) * (norm(wSVR(:,i))^2 - D);
    fprintf("f = %f \n", f);
end 
 toc