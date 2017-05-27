f = @(x) 1 ./ (1 + exp(-x));
df = @(x) f(x) .* (1 - f(x));

w = -1 + 2 * rand(1, 1);
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
for t = 2:T
    net(t) = w * s(t-1) + x(t-1);
    s(t) = f(net(t));
    p(t) = df(net(t)) * (s(t-1) + w * p(t-1));
    w = w - alpha * (s(t) - d(t)) * p(t)
    
    netArr = [netArr; net(t)];
    sArr = [sArr; s(t)];
    pArr = [pArr; p(t)];
    wArr = [wArr; w];
end

plot(netArr, 'r');
hold on;
plot(sArr, 'g');
plot(pArr, 'b');
plot(wArr, 'y');

s(1) = 0; % reset activation of the recurrent unit
T = 10000;
x = (rand(1, T) > 0.2) * 1;
d = [0 1 - x];
pass = 0;
for t = 2:T
    net(t) = w * s(t-1) + x(t-1);
    s(t) = f(net(t));
    if ((s(t) >= 0.5) - d(t)) == 0
    pass = pass + 1;
    end
end
pass