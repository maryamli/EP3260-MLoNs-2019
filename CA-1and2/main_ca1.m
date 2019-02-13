function main_ca1(flagData)
% (MLoNs) Computer Assignment - 1
% Group 3
% DESCRIPTION
% main_ca1(flagData)
% INPUT  (all inputs are optional)
% flagData               = 0; %0 for 'Communities and Crime'  and 1 for 'Individual household'

%% 
% clear variables; % Comment this part if you want to have flagData as input

close all;

clc;

rng(0); 

%% load data (select either of them)

prcntof_data_for_training = 0.8; % for training and the remianing for the validation
normalized_data = 1; % 1 means data is within [-1,1] and 0 means that we need to normalize

if flagData == 0
    load('Communities_Crime/x_data');
    load('Communities_Crime/y_data');     
    
    n       = size(matX_input, 1); % total nr of samples
    d       = size(matX_input, 2); % dimension of the feature vector
    
    n_train = ceil(n * prcntof_data_for_training); % nr of samples for training
    X_train = matX_input(1:n_train, :);
    y_train = y_data(1:n_train); %y_sub_metering_2; y_sub_metering_3;
    
    n_test  = n - n_train;    % nr of test samples
    X_test  = matX_input(n_train+1:end, :);
    y_test  = y_data(n_train+1:end); %y_sub_metering_2; y_sub_metering_3;
    
else
    
    load('Individual_Household/x_data');
%     load('Individual_Household/y_data'); % not normalized in [-1,1]
    load('Individual_Household/y_data_m11'); % normalized in [-1,1]
    
    n       = size(matX_input, 1); % total nr of samples
    d       = size(matX_input, 2); % dimension of the feature vector
    
    n_train = ceil(n * prcntof_data_for_training); % nr of samples for training
    X_train = matX_input(1:n_train, :);
    y_train1= y_sub_metering_1_m11(1:n_train);%y_sub_metering_1(1:n_train); %y_sub_metering_2; y_sub_metering_3;
    
    if normalized_data == 0
        % since y_train > 1. We modify it
        y_train = y_train1;
        y_train(y_train1<=0) = -1;
        y_train(y_train1>0)  = +1;
    else
        y_train = y_train1;
    end
    
    n_test  = n - n_train;    % nr of test samples
    X_test  = matX_input(n_train+1:end, :);
    y_test1 = y_sub_metering_1_m11(n_train+1:end); %y_sub_metering_2; y_sub_metering_3;
    
    if normalized_data == 0
        % since y_train > 1. We modify it
        y_test = y_test1;
        y_test(y_test1<=0) = -1;
        y_test(y_test1>0)  = +1;
    else
        y_test = y_test1;
    end
    
    clear matX_input;

clear matX_input;
    
end



flag_enable_cvx = false; % it is quite slow and do not run in parfor loops: so KEEP it FALSE
if flag_enable_cvx
    run 'cvx_setup.m'
    run 'cvx_startup.m';
end

%% optimal Weights for the linear regression
J_cost_func_lin_reg = @(X, y, N, w, lambda) (1/N) * norm(X*w - y, 2)^2 + lambda*0.5*norm(w,2)^2;

%lambda_vec          = [0, logspace(-20, 0, 20)];
lambda_vec          = [0, logspace(-20, 20, 100)];

w_ls                = zeros(d, numel(lambda_vec));
w_cvx               = zeros(d, numel(lambda_vec));
cost_fun_metric_ls__train   = zeros(1, numel(lambda_vec));
cost_fun_metric_ls__test    = zeros(1, numel(lambda_vec));
cost_fun_metric_cvx__train  = zeros(1, numel(lambda_vec));
cost_fun_metric_cvx__test   = zeros(1, numel(lambda_vec));

if flag_enable_cvx==false
    parfor i_lambda = 1:numel(lambda_vec)
        
        lambda              = lambda_vec(i_lambda);
        w_ls(:, i_lambda)   = (X_train.' * X_train + n_train*lambda*eye(d)) \ (X_train.' * y_train);
        %w_cvx(:, i_lambda)  = run_cvx_for_linear_regression(X_train, y_train, n_train, d, lambda);
        
        cost_fun_metric_ls__train(i_lambda)   = J_cost_func_lin_reg(X_train, y_train, n_train, squeeze(w_ls(:, i_lambda)), lambda);
        %cost_fun_metric_cvx__train(i_lambda)  = J_cost_func_lin_reg(X_train, y_train, n_train, squeeze(w_cvx(:, i_lambda)), lambda);
        
        % do some cross-validation
        cost_fun_metric_ls__test(i_lambda)    = J_cost_func_lin_reg(X_test, y_test, n_test, squeeze(w_ls(:, i_lambda)), lambda);
        %cost_fun_metric_cvx__test(i_lambda)   = J_cost_func_lin_reg(X_test, y_test, n_test, squeeze(w_cvx(:, i_lambda)), lambda);
    end
    
else
    for i_lambda = 1:numel(lambda_vec)
        
        lambda              = lambda_vec(i_lambda);
        tic
        w_ls(:, i_lambda)   = (X_train.' * X_train + n_train*lambda*eye(d)) \ (X_train.' * y_train);
        toc
        tic
        w_cvx(:, i_lambda)  = run_cvx_for_linear_regression(X_train, y_train, n_train, d, lambda);
        toc
        cost_fun_metric_ls__train(i_lambda)   = J_cost_func_lin_reg(X_train, y_train, n_train, squeeze(w_ls(:, i_lambda)), lambda);
        cost_fun_metric_cvx__train(i_lambda)  = J_cost_func_lin_reg(X_train, y_train, n_train, squeeze(w_cvx(:, i_lambda)), lambda);
        
        % do some cross-validation
        cost_fun_metric_ls__test(i_lambda)    = J_cost_func_lin_reg(X_test, y_test, n_test, squeeze(w_ls(:, i_lambda)), lambda);
        cost_fun_metric_cvx__test(i_lambda)   = J_cost_func_lin_reg(X_test, y_test, n_test, squeeze(w_cvx(:, i_lambda)), lambda);
    end
    
    
end


%% Plot
figure(1); clf; 

if flag_enable_cvx==false
    semilogx(lambda_vec, 10*log10(cost_fun_metric_ls__train), 'linewidth', 3); hold on;
    semilogx(lambda_vec, 10*log10(cost_fun_metric_ls__test), 'linewidth', 3, 'linestyle', '--', 'color', 'r');
    xlabel('Lambda');
    ylabel('Cost Function [dB]');
    legend('train', 'test', 'location', 'northwest');
else
    semilogx(lambda_vec, 10*log10(cost_fun_metric_ls__train), 'linewidth', 3); hold on;
    semilogx(lambda_vec, 10*log10(cost_fun_metric_ls__test), 'linewidth', 3, 'linestyle', '--', 'color', 'r');
    semilogx(lambda_vec, 10*log10(cost_fun_metric_cvx__train), 'linewidth', 3, 'linestyle', '--', 'color', 'r');
    semilogx(lambda_vec, 10*log10(cost_fun_metric_cvx__test), 'linewidth', 3, 'linestyle', '--', 'color', 'r');
    xlabel('Lambda');
    ylabel('Cost Function [dB]');
    legend('LS train', 'LS test', 'CVX train', 'CVX test', 'location', 'northwest');
end

end


%%
function w_cvx  = run_cvx_for_linear_regression(X, y, N, d, lambda)

cvx_begin quiet
    variable w_cvx(d,1) 
    minimize ( (1/N) * square_pos(norm(X*w_cvx - y, 2)) + lambda*0.5*square_pos(norm(w_cvx,2)) ) 
    %minimize ( (1/N) * ((X*w_cvx - y)'*(X*w_cvx - y)) + lambda*0.5*(w_cvx'*w_cvx) ) 
cvx_end

end
