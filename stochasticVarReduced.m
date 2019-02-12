function w = stochasticVarReduced(Z, w, alpha, num_iters, lambda, epsilon)
%STOCHASTICAVERAGEGRADIENT Summary of this function goes here
% Initialize some useful values
T = 100;
K = floor(num_iters / T);

N = size(Z,2); % number of training examples
d = size(Z,1);

for k = 1:K
    Gavg = Z * diag(1./(1 + exp(-w'*Z)) - ones(1,N));
    gavg = (1/N) * Gavg * ones(N,1);
    for t = 1:T
        r = randi(N);
        g =  Z(:,r) * (1 / (1 + exp(-w' * Z(:,r))) - 1);
        w = w - alpha * (g - Gavg(:, r) + gavg + 2 * lambda * w);
    end
%     if norm(g) <= epsilon
%         break;
%     end
end

fprintf("norm(g) = %f\n", norm(g));

end

