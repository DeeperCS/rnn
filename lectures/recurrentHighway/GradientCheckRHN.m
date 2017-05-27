clc; clear; close all;
%% activation functions

f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

h = @(x)2*f(x)-1;
dh = @(x)2*df(x);

h = @(x)x;
dh = @(x)1;

f_k = f;  df_k = df;
f_H = f; df_H = df;
f_T = f; df_T = df;
f_out = f;  df_out = df;

delta = @(x,y) x==y;
%% initialization
n = 8; % hidden
m = 7; % input
k = 7; % output

W_in_T = -1 + 2 * rand(n, m+1);
W_in_H = -1 + 2 * rand(n, m+1);
W_R_T = -1 + 2 * rand(n, n+1);
W_R_H = -1 + 2 * rand(n, n+1);
W_kc = -1 + 2 * rand(k, n+1);

gW_in_T = zeros(size(W_in_T));
gW_in_H = zeros(size(W_in_H));
gW_R_T = zeros(size(W_R_T));
gW_R_H = zeros(size(W_R_H));
gW_kc = zeros(size(W_kc));

%% train
alpha = 0.5;

trainLength = 10000;
errArr = [];
% plotHandle = plot(errArr);
for tt = 1:trainLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
%     input = input(1:3, :);
    x = input;
    
    s = {};
    s{1} = zeros(n, 1);
    % Accumulate gradients for every time step    
    gW_in_T = zeros(size(W_in_T));
    gW_in_H = zeros(size(W_in_H));
    gW_R_T = zeros(size(W_R_T));
    gW_R_H = zeros(size(W_R_H));
    gW_kc = zeros(size(W_kc));
    
    errMean = [];
    for t = 2:size(x, 1)
        % forward computation
        d = x(t, :)';
        net_H{t} = W_in_H * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R_H * [s{t-1}; ones([size(s{t-1}, 2)], 1)];
        net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
        s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
        net_k{t} = W_kc * s{t};
        y{t} = f_k(net_k{t});
        y_k = y{t};
        err = 0.5 * (y_k - d)' * (y_k - d);
        errMean = [errMean; err];
    end
    
    delta_next = zeros(n, 1);
    for t = size(x, 1):-1:2
        % compute the gradient of w_kc
        d = x(t, :)';
        delta_K = (y{t} - d) .* df_k(net_k{t});
        gW_kc = gW_kc + delta_K * s{t}' ;
        
        % backpropagation to s
        delta_s = (W_kc' * delta_K) + delta_next;

        gW_in_T = gW_in_T + ( delta_s .* ( df_T(net_T{t}) .* f_H(net_H{t}) - s{t-1} .* df_T(net_T{t}) ) ) * x(t-1, :);
        gW_in_H = gW_in_H + ( delta_s .* f_T(net_T{t}) .* df_H(net_H{t}) ) * x(t-1, :);
        gW_R_T = gW_R_T + ( delta_s .* ( df_T(net_T{t}) .* f_H(net_H{t}) - s{t-1} .* df_T(net_T{t}) ) ) * s{t-1}';
        gW_R_H = gW_R_H + ( delta_s .* f_T(net_T{t}) .* df_H(net_H{t}) ) * s{t-1}';
        % backpropagation through time
        delta_next = ...
            W_R_H' * (delta_s .* df_H(net_H{t}) .* f_T(net_T{t}))  +  ...
            W_R_T' * (delta_s .* f_H(net_H{t}) .* df_T(net_T{t})) + ...
            delta_s .* (1 - f_T(net_T{t})) - ...
            W_R_T' * (delta_s .* s{t-1} .* df_T(net_T{t}));
    end
    
    %% Gradient Check
    s = {};
    s{1} = zeros(n, 1);
    % Accumulate gradients for every time step
    temp = W_kc;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_kc = temp;
            W_kc(i, j) = W_kc(i, j) + epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_kc = temp;
            W_kc(i, j) = W_kc(i, j) - epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_kc_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    W_kc = temp;
    
    temp = W_in_H;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_in_H = temp;
            W_in_H(i, j) = W_in_H(i, j) + epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_in_H = temp;
            W_in_H(i, j) = W_in_H(i, j) - epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_in_H_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    % W_in_T
    temp = W_in_T;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_in_T = temp;
            W_in_T(i, j) = W_in_T(i, j) + epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_in_T = temp;
            W_in_T(i, j) = W_in_T(i, j) - epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_in_T_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    % W_R_T
    temp = W_R_T;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_R_T = temp;
            W_R_T(i, j) = W_R_T(i, j) + epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_R_T = temp;
            W_R_T(i, j) = W_R_T(i, j) - epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_R_T_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    % W_R_T
    temp = W_R_H;
    epsilon = 0.000001;
    for i = 1:size(temp, 1)
        for j = 1:size(temp, 2)
            errMean = [];
            W_R_H = temp;
            W_R_H(i, j) = W_R_H(i, j) + epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E1 = sum(errMean);
            
            errMean = [];
            W_R_H = temp;
            W_R_H(i, j) = W_R_H(i, j) - epsilon;
            for t = 2:size(x, 1)
                % forward computation
                d = x(t, :)';
                net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
                net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
                s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
                net_k = W_kc * s{t};
                y{t} = f_k(net_k);
                y_k = y{t};
                err = 0.5 * (y_k - d)' * (y_k - d);
                errMean = [errMean; err];
            end
            E2 = sum(errMean);
            gW_R_H_Check(i, j) = (E1 - E2) / (2*epsilon);
        end
    end
    
    fprintf('Check gW_R_H:%.10f\n', norm(gW_R_H_Check-gW_R_H));
    fprintf('Check gW_R_T:%.10f\n', norm(gW_R_T_Check-gW_R_T));
    fprintf('Check gW_in_H:%.10f\n', norm(gW_in_H_Check-gW_in_H));
    fprintf('Check gW_in_T:%.10f\n', norm(gW_in_T_Check-gW_in_T));
    fprintf('Check gW_kc:%.10f\n', norm(gW_kc_Check-gW_kc));
    % Gradient Check End
    
    % Update
    W_R_H = W_R_H - alpha * gW_R_H;
    W_R_T = W_R_T - alpha * gW_R_T;
    W_in_H = W_in_H - alpha * gW_in_H;
    W_in_T = W_in_T - alpha * gW_in_T;
    W_kc = W_kc - alpha * gW_kc;
    
    errArr = [errArr; mean(errMean)];
end
plotHandle = plot(errArr);


%% test
%% test
testLength = 10000;
pass = 0;
for tt = 1:testLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
    x = input;
    d = target;
    succeed = 1;
    
    s = {};
    s{1} = zeros(n, 1);
    for t = 2:size(x, 1)
        % forward computation
        net_H{t} = W_in_H * x(t-1, :)' + W_R_H * s{t-1};
        net_T{t} = W_in_T * x(t-1, :)' + W_R_T * s{t-1};
        s{t} = f_H(net_H{t}) .* f_T(net_T{t}) + s{t-1} .* (1 - f_T(net_T{t}));
        net_k = W_kc * s{t};
        y{t} = f_k(net_k);
        y_k = y{t};
        
        [sortedValues, sortedPos] = sort(y_k);
        pred = zeros(size(y_k));
        if str(t-1) == 'E'
            pred(sortedPos(end)) = 1;
        % There's only one target for the second symbol
        elseif (t-1)==2
            pred(sortedPos(end)) = 1;
        % Only one target for the last three symbols (e.g. ETE, EPE)
        elseif t>=(length(str)-2)
            pred(sortedPos(end)) = 1;
        else
            pred(sortedPos(end-1:end)) = 1;
        end
        tar = target(t-1, :);
        pred = pred';
        if sum(abs(tar - pred)) ~= 0 
            succeed = 0;
            break;
        end
    end
    if succeed==1
       pass = pass + 1; 
    end
end
pass