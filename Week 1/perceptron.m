clear;
clc;
pkg load control;

function g = sigmoid(a)
  g = 1/(1 + exp(-a));
endfunction

function g = tanh(a)
  g = (exp(a) - exp(-a))/(exp(a) + exp(-a));
endfunction

N = 20;
learning_rate = 1;
offset = 5;
x = [randn(2,N), randn(2,N)+offset]
y = [zeros(1,N) ones(1,N)];

figure(1)
scatter(1:80, x)

P = x(:,1:N);
Q = x(:,(N+1):N*2);

w = randn(2,1)*0.01;
b = 0;

for i = 1:1000
  r = randi(size(x, 2));
  input = x(:, r);
  [result, _] = ismember(input, P);
  [result1, _] = ismember(input, Q);
  if result == [1;1] && (w'*input + b) >= 0
    w = w - learning_rate*input;
  elseif result1 == [1;1] && (w'*input + b) < 0
    w = w + learning_rate*input;
  endif  
endfor
w = w/20
e = y - w'*x
figure(3)
plot((w'*x))
figure(2)
scatter(x(1, :), y,[],[1 0 0])
hold on
scatter(x(2, :), y,[],[0 0 1])
hold on
plot((w'*x), y)
hold off
figure(4)
plot(e)
