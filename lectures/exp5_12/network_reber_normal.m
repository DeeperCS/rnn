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

% W_I = -1 + 2 * rand(n, m+1);
% W_R = -1 + 2 * rand(n, n+1);
% W_kc = -1 + 2 * rand(k, n+1);

W_I = -1 + 2 * normrnd(0, 0.1, [n, m+1]); 
W_R = -1 + 2 * normrnd(0, 0.1, [n, n+1]); 
W_kc = -1 + 2 * normrnd(0, 0.1, [k, n+1]); 



gW_I = zeros(size(W_I));
gW_R = zeros(size(W_R));
gW_kc = zeros(size(W_kc));

%% train
alpha = 0.1;

trainLength = 20000;
errArr = [];
% plotHandle = plot(errArr);
for tt = 1:trainLength
    % Generate data
    [input, target, str] = reber_gen();
    x = input;
    % T = size(x, 1);
    T = 100;
    s{1} = zeros(n, 1);
    net{1} = zeros(n, 1);
    % Accumulate gradients for every time step    
    gW_I = zeros(size(W_I));
    gW_R = zeros(size(W_R));
    gW_kc = zeros(size(W_kc));
    
    errMean = [];
    for t = 2:size(x, 1)
        % forward computation
        d = x(t, :)';
        net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
        s{t} = s{t-1} + ( f(net{t}) - s{t-1} ) .* (cos(t/T)^2);

        % Using the last layer as output
        net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
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
        gW_kc = gW_kc + delta_K * [s{t}; ones([size(s{t}, 2), 1])]';
        
        % backpropagation to s
        delta_s{t} = (W_kc(:, 1:end-1)' * delta_K) + delta_next;
        
        gW_R = gW_R + ( delta_s{t} .* cos(t/T)^2 .* df(net{t}) ) * [s{t-1}; ones([size(s{t-1}, 2), 1])]';
        gW_I = gW_I + ( delta_s{t} .* cos(t/T)^2 .* df(net{t}) ) * [x(t-1, :), ones([size(x(t-1, :), 1), 1])];
        
        % delta_s{t-1} = delta_s{t} .* ( 1 + cos(t/T)^2 .* (df(net{t}) - 1) );
        delta_s{t-1} = delta_s{t} + W_R(:, 1:end-1)' * ( delta_s{t} .* cos(t/T)^2 .* df(net{t}) ) - delta_s{t} .* cos(t/T)^2;
       
        delta_next = delta_s{t-1};
    end
    
    % Update
    W_I = W_I - alpha * gW_I;
    W_R = W_R - alpha * gW_R;
    W_kc = W_kc - alpha * gW_kc;

    errArr = [errArr; sum(errMean)];
end
plotHandle = plot(errArr);

%% test
testLength = 10000;
pass = 0;
for tt = 1:testLength
    % Generate data
    [input, target, str] = reber_gen();
    x = input;
    % T = size(x, 1);
    d = target;
    succeed = 1;
    
    s = {};
    s{1} = zeros(n, 1);
    for t = 2:size(x, 1)
        % forward computation
        d = x(t, :)';
        net{t} = W_I * [x(t-1, :), ones([size(x(t-1, :), 1), 1])]' + W_R * [s{t-1}; ones([size(s{t-1}, 2), 1])];
        s{t} = s{t-1} + ( f(net{t}) - s{t-1} ) .* (cos(t/T)^2);

        % Using the last layer as output
        net_k{t} = W_kc * [s{t}; ones([size(s{t}, 2), 1])];
        y{t} = f_k(net_k{t});
        y_k = y{t};
        
        [sortedValues, sortedPos] = sort(y_k);
        pred = zeros(size(y_k));
        if str(t) == 'E'
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