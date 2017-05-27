f = @(x) 1 ./ (1 + exp(-x));
df = @(x) f(x) .* (1 - f(x));

w_cc = -1 + 2 * rand(1, 1);
w_kc = -1 + 2 * rand(1, 1);

s(1) = 0;
p(1) = 0;

alpha = 0.2;
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

for t = 2:T
%     Feed Forward
    net_c(t) = w_cc * s(t-1) + x(t-1);
    s(t) = s(t-1) + f(net_c(t));
    net_k(t) = w_kc * s(t);
    y_k(t) = f(net_k(t));
    
    delta_k(t) = ( y_k(t) - d(t) ) * df(net_k(t));
%     Update w_kc
    w_kc = w_kc - alpha * delta_k(t) * s(t);
    
    delta_c(t) = w_kc * delta_k(t);
    p(t) = p(t-1) + df(net_c(t)) * s(t-1);
    
    w_cc = w_cc - alpha * delta_c(t) * p(t);
    
    w_kc_Arr = [w_kc_Arr; w_kc];
    w_cc_Arr = [w_cc_Arr; w_kc];
    
    yArr = [yArr; y_k(t)];
%     netArr = [netArr; net(t)];
%     sArr = [sArr; s(t)];
%     pArr = [pArr; p(t)];
%     wArr = [wArr; w];
end

plot(yArr, 'r');
plot(w_kc_Arr, 'r');
plot(w_cc_Arr, 'r');
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
    net_c(t) = w_cc * s(t-1) + x(t-1);
    s(t) = s(t-1) + f(net_c(t));
    net_k(t) = w_kc * s(t);
    y_k(t) = f(net_k(t));
    if ((y_k(t) >= 0.5) - d(t)) == 0
        pass = pass + 1;
    end
end
pass