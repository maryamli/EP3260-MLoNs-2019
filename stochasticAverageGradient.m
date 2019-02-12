function w = stochasticAverageGradient(Z, w, alpha, num_iters, lambda, epsilon)
%STOCHASTICAVERAGEGRADIENT Summary of this function goes here
% Initialize some useful values
N = size(Z,2); % number of training examples
d = size(Z,1);
G = zeros(d,N);

for iter = 1:num_iters
    r = randi(N);
    G(:,r) =  Z(:,r) * (1 / (1 + exp(-w' * Z(:,r))) - 1);
    g = (1/N) * G * ones(N,1) + 2 * lambda * w;
    w = w - alpha * g;
%     if norm(g) <= epsilon
%         break;
%     end
end

fprintf("norm(g) = %f\n", norm(g));

end

