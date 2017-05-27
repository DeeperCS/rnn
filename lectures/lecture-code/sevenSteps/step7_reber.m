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

p = zeros(n, size(w_1, 1), size(w_1, 2));

s_c = zeros(n, 1);
y_c = zeros(n, 1);
%% train
alpha = 1;

trainLength = 2000;
errArr = [];
plotHandle = plot(errArr);
for tt = 1:trainLength
    % Generate data
    [input, target, str] = reber_gen();
    x = input;
    d = target;

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
        errArr = [errArr; err];
        delete(plotHandle);
        plotHandle = plot(errArr);
        pause(0.001);
        %% error
        dK = zeros(k, 1);
        dSc = zeros(n, 1);
        dOut = zeros(n, 1);

        % compute the gradient of w_kc
        dK = (y_k - x(t+1, :)') .* df_k(net_k);
        gW_kc = dK * y_c' ;

        % backpropagation to dOut and dSc
        dOut = (w_kc' * dK) .* h(s_c) .* df_out(net_out);
        dSc = (w_kc' * dK) .* dh(s_c) .* f_out(net_out);

        % compute the gradient of w_out
        gWout = dOut * y_u';

        %% Computing P
        for dp = 1:n
            for di = 1:size(w_1, 1)
                for dj = 1:size(w_1, 2)
                    p(dp, di, dj) = p(dp, di, dj) + df1(net_1(di)) * f2(net_2(di)) * ( delta(dp, di) * y_u(dj) );
                end
            end
        end

        % gW1, gW2
        for dj = 1:size(gW1, 2)
            gW1(:, dj) = p(1, :, dj) * dSc(:);
            gW2(:, dj) = p(2, :, dj) * dSc(:);
        end

        % update weights
        w_1 = w_1 - alpha * gW1;
        w_2 = w_2 - alpha * gW2;
        w_out = w_out - alpha * gWout;
        w_kc = w_kc - alpha * gW_kc;
    end
end

%% test
% s_c{1} = zeros(n, 1);
% y_c{1} = zeros(n, 1);

testLength = 1000

for tt = 1:testLength
    % Generate data
    [input, target, str] = reber_gen();
    x = input;
    d = target;
    succeed = 1;
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
        y_k'
        d(t,:)
        
        err = 0.5 * sum(y_k - x(t+1, :)')^2;
    end
end

