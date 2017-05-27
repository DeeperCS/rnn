f = @(x) 1 ./ (1 + exp(-x));
df = @(x) f(x) .* (1 - f(x));
h = @(x) 2 .* f(x) - 1;
dh = @(x) 2 .* df(x);
f_k = f;
df_k = df;

w_cc = -1 + 2 * rand(1, 1);
w_cx = -1 + 2 * rand(1, 1);
w_kc = -1 + 2 * rand(1, 1);

s_c(1) = 0;
p_c(1) = 0;
p_x(1) = 0;
y_c(t) = 0;

alpha = 1;
T = 10000;
x = (rand(1, T) > 0.2) * 1;
d = [0 1-x];

netArr = [];
sArr = [];
pArr = [];
wArr = [];
yArr = [];
w_kc_Arr = [];
w_cc_Arr = [];
w_cx_Arr = [];

for t = 2:T
%     Feed Forward
    net_c(t) = w_cc * y_c(t-1) + w_cx * x(t-1);
    s_c(t) = s_c(t-1) + f(net_c(t));
    y_c(t) = h(s_c(t));
    
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
%     Update 
    delta_k(t) = ( y_k(t) - d(t) ) * df_k( net_k(t) );
    
    dw_kc = delta_k(t) * y_c(t);
    
    delta_c(t) = w_kc * delta_k(t) * dh(s_c(t));
    
    p_c(t) = p_c(t-1) + df(net_c(t)) * y_c(t-1);
    p_x(t) = p_x(t-1) + df(net_c(t)) * x(t-1);
    
    dw_cc = delta_c(t) * p_c(t);
    dw_cx = delta_c(t) * p_x(t);
    
    w_cc = w_cc - alpha * dw_cc;
    w_cx = w_cx - alpha * dw_cx;
    w_kc = w_kc - alpha * dw_kc;
    
    w_kc_Arr = [w_kc_Arr; w_kc];
    w_cc_Arr = [w_cc_Arr; w_kc];
    w_cx_Arr = [w_cx_Arr; w_cx];
    
    yArr = [yArr; y_k(t)];
%     netArr = [netArr; net(t)];
%     sArr = [sArr; s(t)];
%     pArr = [pArr; p(t)];
%     wArr = [wArr; w];
end

% plot(yArr, 'r');
plot(w_cx_Arr, 'r');
% plot(w_kc_Arr, 'r');
% plot(w_cc_Arr, 'r');
% hold on;
% plot(sArr, 'g');
% plot(pArr, 'b');
% plot(wArr, 'y');

s(1) = 0; % reset activation of the recurrent unit
T = 10000;
x = (rand(1, T) > 0.2) * 1;
d = [0 1 - x];
pass = 0;
for t = 2:T
    net_c(t) = w_cc * y_c(t-1) + w_cx * x(t-1);
    s_c(t) = s_c(t-1) + f(net_c(t));
    y_c(t) = h(s_c(t));
    
    net_k(t) = w_kc * y_c(t);
    y_k(t) = f_k(net_k(t));
    
    if ((y_k(t) >= 0.5) - d(t)) == 0
        pass = pass + 1;
    end
end
pass