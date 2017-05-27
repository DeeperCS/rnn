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
n = 8; % hidden
m = 7; % input
k = 7; % output

w_1 = -1 + 2 * rand(n, n+m);
w_2 = -1 + 2 * rand(n, n+m);
w_out = -1 + 2 * rand(n, n+m);
w_kc = -1 + 2 * rand(k, n);

gW1 = zeros(size(w_1));
gW2 = zeros(size(w_2));
gWout = zeros(size(w_out));
gW_kc = zeros(size(w_kc));

%% train
alpha = 1;

trainLength = 5000;
errArr = [];
% plotHandle = plot(errArr);
for tt = 1:trainLength
    % Generate data
    [input, target, str] = reber_gen();
    x = input;
    
    s_c = zeros(n, 1);
    y_c = zeros(n, 1);
    
    p1 = zeros(size(w_1, 1), size(w_1, 2));
    p2 = zeros(size(w_2, 1), size(w_2 , 2));
    
    gW1 = zeros(size(w_1));
    gW2 = zeros(size(w_2));
    
    p_1 = zeros(n, n+m);
    p_2 = zeros(n, n+m);
    
    errMean = [];
    for t = 1:size(x, 1)-1
        % forward computation
        y_u = [y_c; x(t, :)'];
        net_1 = w_1 * y_u;
        net_2 = w_2 * y_u;
        net_out = w_out * y_u;
        s_c = s_c + f1(net_1) .* f2(net_2);
        y_c = h(s_c) .* f_out(net_out);
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

        % backpropagation to dOut and dSc
        dOut = (w_kc' * dK) .* h(s_c) .* df_out(net_out);
        dSc = (w_kc' * dK) .* dh(s_c) .* f_out(net_out);

        % compute the gradient of w_out
        gWout = dOut * y_u';

        %% Computing P (loop)
        p1 = p1 + df1(net_1) .* f2(net_2) * y_u';
        p2 = p2 + f1(net_1) .* df2(net_2) * y_u';

        gW1 = p1 .* repmat(dSc, 1, size(p1, 2));
        gW2 = p2 .* repmat(dSc, 1, size(p2, 2));

%%     %%%%%%%   Computing P (Vector)   %%%%%%%
%         p_1 = p_1 + (f2(net_2) .* df1(net_1)) * y_u';
%         p_2 = p_2 + (f1(net_1) .* df2(net_2)) * y_u';
%         % compute gw_1 and gw_2
%         gW1 = p_1 .* repmat(dSc, 1, n+m);
%         gW2 = p_2 .* repmat(dSc, 1, n+m);
%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update weights
        w_1 = w_1 - alpha * gW1;
        w_2 = w_2 - alpha * gW2;
        w_out = w_out - alpha * gWout;
        w_kc = w_kc - alpha * gW_kc; 
    end
    errArr = [errArr; mean(errMean)];
end
plotHandle = plot(errArr);


%% test
testLength = 10000;
pass = 0;
for tt = 1:testLength
    % Generate data
    [input, target, str] = reber_gen();
    x = input;
    d = target;
    succeed = 1;
    s_c = zeros(n, 1);
    y_c = zeros(n, 1);
    y_k = {};
    for t = 1:size(x, 1)-1
        % forward computation
        y_u = [y_c; x(t, :)'];
        net_1 = w_1 * y_u;
        net_2 = w_2 * y_u;
        net_out = w_out * y_u;
        s_c = s_c + f1(net_1) .* f2(net_2);
        y_c = h(s_c) .* f_out(net_out);
        net_k = w_kc * y_c;
        y_k{t} = f_k(net_k);

        [sortedValues, sortedPos] = sort(y_k{t});
        pred = zeros(size(y_k{t}));
        if str(t+1) == 'E'
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