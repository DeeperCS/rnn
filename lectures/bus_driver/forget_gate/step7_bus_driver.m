clc; clear; close all;
%% activation functions
% 6.694 s
f = @(x) 1 ./ (1 + exp(-x)); % sigmoid
df = @(x) f(x) .* (1 - f(x));

h = @(x)2*f(x)-1;
dh = @(x)2*df(x);

% h = @(x)x;
% dh = @(x)1;
% 
% h = f;
% dh = df;

f_k = f;  df_k = df;
% f1 = h;  df1 = dh;
% f1 = @(x)2*f(x)-1;
% df1 = @(x)2*df(x);
f1 = f;  df1 = df;
f2 = @(x)2*f(x)-1;  df2 = @(x)2*df(x);
f_out = f;  df_out = df;
f_f = f; df_f = df;


delta = @(x,y) x==y;
%% initialization
n = 2; % hidden
m = 4; % input
k = 1; % output

w_1 = -1 + 2 * rand(n, n+m);
w_2 = -1 + 2 * rand(n, n+m);
w_f = -1 + 2 * rand(n, n+m);
w_out = -1 + 2 * rand(n, n+m);
w_kc = -1 + 2 * rand(k, n);

gW1 = zeros(size(w_1));
gW2 = zeros(size(w_2));
gWf = zeros(size(w_f));
gWout = zeros(size(w_out));
gW_kc = zeros(size(w_kc));

% Feed forward
s_c = zeros(n, 1);
y_c = zeros(n, 1);

p1 = zeros(size(w_1));
p2 = zeros(size(w_2));
p_f = zeros(size(w_f));
    
%% train
alpha = 1;

a = [1,0,0,0];
b = [0,1,0,0];
c = [0,0,1,0];
d = [0,0,0,1];

alphabet = [a;b;c;d];
str_alphabet = ['a','b','c','d'];
x_seq = [];
t_seq = [];

trainLength = 30000;
errArr = [];
plotHandle = plot(errArr);
first_a = 0; 
for tt = 1:trainLength
    % Generate data
    idx = randi(4);
    x = alphabet(idx, :);
    followed_b = 0;
    if isequal(x, a)
        first_a = 1;
    elseif isequal(x, b) & first_a == 1
        followed_b = 1;
        first_a = 0;
    end
    x_seq = [x_seq, str_alphabet(idx)];
    t_seq = [t_seq, followed_b];
    target = followed_b;

    % forward computation
    y_u = [y_c; x'];
    net_1 = w_1 * y_u;
    net_2 = w_2 * y_u;
    net_f = w_f * y_u;
    net_out = w_out * y_u;
    s_c_t = s_c;
    s_c = s_c .* f_f(net_f) + f1(net_1) .* f2(net_2);
    y_c = h(s_c) .* f_out(net_out);
    net_k = w_kc * y_c;
    y_k = f_k(net_k);
%     if target==1
%        disp('1') 
%     end
    [y_k, target];
    err = 0.5 * (y_k - target)' * (y_k - target);
    
    if mod(tt,  800) == 0
        delete(plotHandle);
        plotHandle = plot(errArr);
        pause(0.001);
    end
    %% error

    % compute the gradient of w_kc
    dK = (y_k - target) .* df_k(net_k);
    gW_kc = dK * y_c' ;

    % backpropagation to dOut and dSc
    dOut = (w_kc' * dK) .* h(s_c) .* df_out(net_out);
    dSc = (w_kc' * dK) .* dh(s_c) .* f_out(net_out);

    % compute the gradient of w_out
    gWout = dOut * y_u';

    %% Computing P (loop)
    p1 = p1 .* repmat(f_f(net_f), [1, n+m]) + (df1(net_1) .* f2(net_2)) * y_u';
    p2 = p2 .* repmat(f_f(net_f), [1, n+m]) + (f1(net_1) .* df2(net_2)) * y_u';
    p_f = p_f .* repmat(f_f(net_f), [1, n+m]) + (s_c_t .* df_f(net_f)) * y_u';
    gW1 = p1 .* repmat(dSc, 1, m+n);
    gW2 = p2 .* repmat(dSc, 1, m+n);
    gWf = p_f .* repmat(dSc, 1, m+n);
%%    % update weights
    w_1 = w_1 - alpha * gW1;
    w_2 = w_2 - alpha * gW2;
    w_f = w_f - alpha * gWf;
    w_out = w_out - alpha * gWout;
    w_kc = w_kc - alpha * gW_kc; 

    errArr = [errArr; err];
end
plotHandle = plot(errArr);
s_c

%% test
testLength = 20000;
pass = 0;
first_a = 0; 
for tt = 1:testLength
    % Generate data
    idx = randi(4);
    x = alphabet(idx, :);
    followed_b = 0;
    if isequal(x, a)
        first_a = 1;
    elseif isequal(x, b) & first_a == 1
        followed_b = 1;
        first_a = 0;
    end
    x_seq = [x_seq, str_alphabet(idx)];
    t_seq = [t_seq, followed_b];
    target = followed_b;
    % forward computation
    y_u = [y_c; x'];
    net_1 = w_1 * y_u;
    net_2 = w_2 * y_u;
    net_f = w_f * y_u;
    net_out = w_out * y_u;
    s_c_t = s_c;
    s_c = s_c .* f_f(net_f) + f1(net_1) .* f2(net_2);
    y_c = h(s_c) .* f_out(net_out);
    net_k = w_kc * y_c;
    y_k = f_k(net_k);

    err = 0.5 * (y_k - target)' * (y_k - target);
    if (y_k>0.5) - target==0
        pass = pass + 1;
    end

end
pass