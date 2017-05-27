clc; clear; close all;
%% activation functions
% 6.694 s
f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

h = @(x)2*f(x)-1;
dh = @(x)2*df(x);

h = @(x)x;
dh = @(x)1;

f_k = f;  df_k = df;
f1 = f;  df1 = df;
f2 = f;  df2 = df;
f_out = f;  df_out = df;

delta = @(x,y) x==y;
%% initialization
n = 10; % hidden
m = 7; % input
k = 7; % output

w = -1 + 2 * rand(n, n+m);
w_kc = -1 + 2 * rand(k, n);

z_diag = zeros(n, n * (n+m));
for i = 1:n
    z_diag(i, ((i-1)*(n+m)+1):i*(n+m)) = 1;
end

%% train
alpha = 1;

trainLength = 2000;
errArr = [];
plotHandle = plot(errArr);
for tt = 1:trainLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
    x = input;
    
    s_c = zeros(n, 1);
    y_c = zeros(n, 1);
    
    p_v = zeros(n, n * (n+m));
    
    gW = zeros(size(w));
    gW_kc = zeros(size(w_kc));

    errMean = [];
    for t = 1:size(x, 1)-1
        % forward computation
        z = [y_c; x(t, :)'];
        net_1 = w * z;
        y_c = f(net_1);
        net_k = w_kc * y_c;
        y_k = f_k(net_k);

        err = 0.5 * (y_k - x(t+1, :)')' * (y_k - x(t+1, :)');
        errMean = [errMean; err];
%         delete(plotHandle);
%         plotHandle = plot(errArr);
%         pause(0.001);
        %% error

        % compute the gradient of w_kc
        dK = (y_k - x(t+1, :)') .* df_k(net_k);
        gW_kc = dK * y_c' ;

        % backpropagation dSc
        dC = (w_kc' * dK) .* df(net_1);

       %% Computing P (loop)
%         for k = 1:n
%             for i = 1:n
%                 for j = 1:n+m
%                     q1(k,i,j) = df(net_1(k)) * (w_1(k, 1:n) * p1(:, i, j) + delta(k,i) * z(j));
%                 end
%             end
%         end
%         p1 = q1;
% 
%         gW1 = squeeze(sum(p1 .* repmat(dC, [1, n, n+m]), 1));
       p_v = repmat(df(net_1), [1, n*(n+m)]) .* (w(:, 1:n) * p_v + z_diag .* repmat(z', [n, n]));
       gW = reshape(p_v' * dC, [n+m, n])';

%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update weights
        w = w - alpha * gW;
        w_kc = w_kc - alpha * gW_kc; 
    end
    errArr = [errArr; mean(errMean)];
    if mod(tt, 50) == 0
        delete(plotHandle);
        plotHandle = plot(errArr);
        pause(0.0001);
    end
end



%% test
testLength = 10000;
pass = 0;
for tt = 1:testLength
    % Generate data
    [input, target, str] = embeded_reber_gen();
    x = input;
    d = target;
    succeed = 1;
    s_c = zeros(n, 1);
    y_c = zeros(n, 1);
    y_k = {};
    for t = 1:size(x, 1)-1
        % forward computation
        z = [y_c; x(t, :)'];
        net_1 = w * z;
        y_c = f(net_1);
        net_k = w_kc * y_c;
        y_k{t} = f_k(net_k);

        [sortedValues, sortedPos] = sort(y_k{t});
        pred = zeros(size(y_k{t}));
        if str(t+1) == 'E'
            pred(sortedPos(end)) = 1;
        % There's only one target for the second symbol
        elseif t==2
            pred(sortedPos(end)) = 1;
        % Only one target for the last three symbols (e.g. ETE, EPE)
        elseif t>=(length(str)-2)
            pred(sortedPos(end)) = 1;
        else
            pred(sortedPos(end-1:end)) = 1;
        end
        tar = target(t, :);
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