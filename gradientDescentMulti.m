function w = gradientDescentMulti(Z, w, alpha, num_iters, lambda, epsilon)
%GRADIENTDESCENTMULTI Performs gradient descent to learn w

% Initialize some useful values
N = size(Z,2); % number of training examples

for iter = 1:num_iters
    g = (1/N) * Z * diag(1./(1+exp(-w'*Z))-ones(1,N)) * ones(N,1) + 2 * lambda * w;
    w = w - alpha * g;
    if norm(g) <= epsilon
        fprintf("<epsilon norm(g) = %f\n", norm(g));
        break;
    end
end

fprintf("norm(g) = %f\n", norm(g));

end
