close all;
T = 1;
f = @(t) cos(t/T).^2;
f = @(t) cos(t/T);
x = linspace(-10, 10, 100);
y = f(x);
plot(x, y)