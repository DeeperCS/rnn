clc; clear all; close all;
%% activation functions
f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

g = @(x)4*f(x)-2;
dg = @(x)4*df(x);

h = @(x)2*f(x)-1;
dh = @(x)2*df(x);

f_k = f;  df_k = df;
f1 = f;  df1 = df;
f2 = g;  df2 = dg;
f_out = f;  df_out = df;

delta = @(x,y) x==y;
%% initialization
n = 3; % hidden
m = 2; % input
k = 1; % output

w_1 = -1 + 2 * rand(n, n+m);
w_2 = -1 + 2 * rand(n, n+m);
w_out = -1 + 2 * rand(n, n+m);
w_kc = -1 + 2 * rand(k, n);

gW1 = zeros(size(w_1));
gW2 = zeros(size(w_2));
gWout = zeros(size(w_out));
gW_kc = zeros(size(w_kc));

p{1} = zeros(n, size(w_1, 1), size(w_1, 2));

s_c{1} = zeros(n, 1);
y_c{1} = zeros(n, 1);
%% train
alpha = 0.2;
T = 6000;
x = (rand(2, T) > 0.5) * 1;
d = [0, xor(x(1, :), x(2, :))];
errArr = [];
plotHandle = plot(errArr);
for t = 2:T
    % forward computation
    y_u{t-1} = [y_c{t-1}; x(:, t-1)];
    
    net_1{t} = w_1 * y_u{t-1};
    net_2{t} = w_2 * y_u{t-1};
    net_out{t} = w_out * y_u{t-1};

    s_c{t} = s_c{t-1} + f1(net_1{t}) .* f2(net_2{t});
    
    y_c{t} = h(s_c{t}) .* f_out(net_out{t});
    
    net_k{t} = w_kc * y_c{t};
    y_k{t} = f_k(net_k{t});
    
    err = 0.5 * (y_k{t} - d(t))^2;
    errArr = [errArr; err];
    delete(plotHandle);
    plotHandle = plot(errArr);
    pause(0.001);
    %% error
    dK = zeros(k, 1);
    dSc = zeros(n, 1);
    dOut = zeros(n, 1);
    
    % compute the gradient of w_kc
    dK = (y_k{t} - d(t)) * df_k(net_k{t});
    for di = 1:size(gW_kc, 1)
        for dj = 1:size(gW_kc, 2)
            gW_kc(di, dj) = y_c{t}(dj) * dK(di);
        end
    end
    
    % backpropagation to dOut and dSc
    for dj = 1:size(w_kc, 2)
        for di = 1:size(w_kc, 1)
            dSc(dj) = dSc(dj) + w_kc(di, dj) * dK(di);
        end
    end
    dOut = dSc .* h(s_c{t}) .* df_out(net_out{t});
    dSc = dSc .* dh(s_c{t}) .* f_out(net_out{t});
    
    % compute the gradient of w_out
    for di = 1:size(w_out, 1)
        for dj = 1:size(w_out, 2)
            gWout(di, dj) = dOut(di) * y_u{t-1}(dj);
        end
    end
    
    %% Computing P
    for dp = 1:n
        for di = 1:size(w_1, 1)
            for dj = 1:size(w_1, 2)
                p{t}(dp, di, dj) = p{t-1}(dp, di, dj) + df1(net_1{t}(di)) * f2(net_2{t}(di)) * ( delta(dp, di) * y_u{t-1}(dj) );
            end
        end
    end
    
    % gW1, gW2
    for di = 1:size(gW1, 1)
        for dj = 1:size(gW1, 2)
            gW1(di, dj) = p{t}(1, di, dj) * dSc(di);
            gW2(di, dj) = p{t}(2, di, dj) * dSc(di);
        end
    end

    % update weights
    w_1 = w_1 - alpha * gW1;
    w_2 = w_2 - alpha * gW2;
    w_out = w_out - alpha * gWout;
    w_kc = w_kc - alpha * gW_kc;
end

%% test
% s_c{1} = zeros(n, 1);
% y_c{1} = zeros(n, 1);

T = 1000;
x = (rand(2, T) > 0.5) * 1;
d = [0, xor(x(1, :), x(2, :))]; 

pass = 0;
for t = 2:T
    % forward computation
    y_u{t-1} = [y_c{t-1}; x(:, t-1)];
    net_1{t} = w_1 * y_u{t-1};
    net_2{t} = w_2 * y_u{t-1};
    net_out{t} = w_out * y_u{t-1};
    s_c{t} = s_c{t-1} + f1(net_1{t}) .* f2(net_2{t});
    y_c{t} = h(s_c{t}) .* f_out(net_out{t});
    net_k{t} = w_kc * y_c{t};
    y_k{t} = f_k(net_k{t});
    
    if ((y_k{t} >= 0.5) - d(t)) == 0
        pass = pass + 1;
    end
end
pass