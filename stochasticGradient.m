function w = stochasticGradient(Z, w, alpha, num_iters, lambda, epsilon)
% Initialize some useful values
N = size(Z,2); % number of training examples

for iter = 1:num_iters
    r = randi(N);
    g = Z(:,r) * (1 / (1 + exp(-w' * Z(:,r))) - 1) + 2 * lambda * w;
    w = w - alpha * g;
%     if norm(g) <= epsilon
%         break;
%     end
end

fprintf("norm(g) = %f\n", norm(g));

end

