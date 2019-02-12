%% 2
X = load('X.txt');
y = load('Y.txt');
lambda = 0.1;
d = size(X,1);
N = size(X,2);
w = (X * X' + lambda * eye(d)) \ (X * y);

%% 3
clear all;
data = load('household_v3.txt');
lambda = 0.1;
X = data(:,1:6)';
y = data(:,9);
N = length(y);
d = size(X,1);
w = (X * X' + lambda * eye(d)) \ (X * y);
