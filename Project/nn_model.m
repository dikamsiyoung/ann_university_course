function parameters = nn_model(n_h, num_iterations, print_cost, learning_rate ) %n_h is number of hidden layer units, print_cost is a boolean
    dataset = csvread('planar_data.csv');
    X = dataset(2:3, 2:end);
    Y = dataset(4, 2:end);
    [n_x, n_y] = layer_sizes(X,Y);
    layers = [5, 10, 20, 50];
    
    for layer = 1:length(layers)
    n_h = layers(layer);
    parameters = initialize_parameters(n_x, n_h, n_y);
    for i = 1:num_iterations
        [A2, cache] = forward_propagation(X, parameters);
        
        cost = compute_cost(A2, Y, parameters, learning_rate);
        
        grads = backward_propagation(parameters, cache, X,Y, learning_rate);
        
        parameters = update_parameters(parameters, grads);
        
        if (print_cost && rem(i,500) == 0)
            % disp(["Cost after iteration", i,":", cost])
            %fprintf('Cost after iteration %i: %f\n',i,cost)
        endif
    endfor
    W1 = parameters("W1");
    b1 = parameters("b1");
    W2 = parameters("W2");
    b2 = parameters("b2");
    fprintf("\n")
    disp('-----------------------------------------------')
    predictions = predict(parameters, X);
    fprintf('Number of hidden layer units: %i\n',n_h)
    accuracy = accuracy_score(predictions, Y)
    fprintf('Number of iterations: %i\n',num_iterations)
    fprintf('Learning rate: %i\n',learning_rate)
    fprintf('-------------------------------------------------\n')
    endfor 
endfunction

function [X, Y] = load_planar_dataset
  t1 = linspace(0, 3.12, 200) + (randn(1, 200) * 0.2);
  t2 = linspace(1, 6.24, 200) + (randn(1, 200) * 0.2); 
  r1 = (4 * sin(4 * t1)) + (randn(1, 200) * 0.2); 
  r2 = (4 * sin(4 * t2)) + (randn(1, 200) * 0.2); 
  f1 = cat(1, (r1 .* sin(t1)), (r1 .* cos(t1))); 
  f2 = cat(1, (r2 .* sin(t2)), (r2 .* cos(t2))); 
  c = cat(2, f1, f2);
  figure; hold on;
  scatter(c(1,1:200),c(2, 1:200), 'x');
  scatter(c(1,201:400),c(2, 201:400), 'o');
  hold off;
  X = c;
  y = ones(1, 200); 
  x = zeros(1, 200);
  Y = cat(2, x, y);
endfunction

function [n_x,n_y]=layer_sizes(X,Y)
n_x = size(X, 1);
n_y = size(Y, 1);
endfunction

function [parameters]=initialize_parameters(n_x,n_h,n_y)
  W1=randn(n_h,n_x) * 2*0.01 - 0.01;
  b1=zeros(n_h,1);
  W2=randn(n_y,n_h) * 2*0.01 - 0.01;
  b2=zeros(n_y,1);

  assert(size(W1) == [n_h, n_x]);
  assert(size(b1) == [n_h, 1]);
  assert(size(W2) == [n_y, n_h]);
  assert(size(b2) == [n_y, 1]);
  keys={'W1','b1','W2','b2'};
  values={W1,b1,W2,b2};
  parameters=containers.Map(keys,values);
end

function [A2, cache]= forward_propagation(X, parameters)
W1=parameters('W1');
b1=parameters('b1');
W2=parameters('W2');
b2=parameters('b2');

Z1=(W1*X) + b1;
A1=sigmoid(Z1);
Z2=(W2*A1) + b2;
A2=sigmoid(Z2);

keys={'Z1','A1','Z2','A2'};
values={Z1,A1,Z2,A2};
cache=containers.Map(keys,values);
end

function[g]=sigmoid(z)
g=1./(1+exp(-z));
end

function cost=compute_cost(A2, Y, parameters, learning_rate)
    m=size(Y);
    m=m(2);
    logprobs=(log(A2).*(Y)) + log(1-A2).*(1-Y);
    cost = -(1/m) * learning_rate.*(sum(logprobs));
    
    cost=double(squeeze(cost));
end

function grads = backward_propagation(parameters, cache, X, Y, learning_rate)
    m=size(X);
    m=m(1);
    
    W1 = parameters('W1');
    W2 = parameters('W2');
    
    A1 = cache('A1');
    A2 = cache('A2');
    
    dZ2 = A2-Y;
    dW2 = (1/m) * learning_rate.*(dZ2 * A1');
    db2 = (1/m) * learning_rate.*(sum(dZ2,2));
    dZ1 = (W2' * dZ2) .* (1-(A1.^2));
    dW1 = (1/m) * learning_rate.*(dZ1 * X');
    db1 = (1/m) * learning_rate.*(sum(dZ1,2));    
    
    keys = {'dW1', 'db1', 'dW2', 'db2'};
    values = {dW1, db1, dW2, db2};
    grads = containers.Map(keys, values);
end

function parameters = update_parameters(parameters, grads, learning_rate)
    learning_rate=1.2;
    W1 = parameters("W1");
    b1 = parameters("b1");
    W2 = parameters("W2");
    b2 = parameters("b2");
    
    dW1 = grads("dW1");
    db1 = grads("db1");
    dW2 = grads("dW2");
    db2 = grads("db2");
    
    W1 = W1 - (learning_rate * dW1);
    b1 = b1 - (learning_rate * db1);
    W2 = W2 - (learning_rate * dW2);
    b2 = b2 - (learning_rate * db2);
    
    keys = {'W1', 'b1', 'W2', 'b2'};
    values = {W1, b1, W2, b2};
    parameters = containers.Map(keys, values);
    
end

function predictions = predict(parameters, X)
    [A2, Cache] = forward_propagation(X, parameters);
    predictions = (A2 > 0.5);
end

function accuracy = accuracy_score(predictions, y)
  correct = 0;
  m = length(y);
  for i = 1:m
    if predictions(i) == y(i)
      correct = correct + 1;
    endif
  endfor
  accuracy = (correct/m) * 100;
endfunction
