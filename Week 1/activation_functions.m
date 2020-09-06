#This file is created by Dikamsi Young Udochi, 15CJ02885

z = [];

#Identity Function
function output = identity_func(z)
  output = z;
endfunction

#Step Function: Binary
function output = binary_step_func(z, T)
  output  = 0;
  if (z >= T)
    output = 1;
  else
    output = 0;
  endif
endfunction

#Step Function: Bipolar
function output = bipolar_step_func(z, T)
  output = 0;
  if (z >= T)
    output = 1;
  else
    output = -1;
  endif
endfunction

#Sigmoid Function: Binary
function output = binary_sigmoid_func(z, alpha)
  output = 1/(1 + exp(-alpha * z));
endfunction

#Sigmoid Function: Bipolar
function output = bipolar_sigmoid_func(z, alpha)
  output = (1 - exp(-alpha * z))/(1 + exp(-alpha * z));
endfunction

#Softmax Function
function output = softmax_func(z)
  output = exp(z)/sum(exp(z));
endfunction

#ReLU Function
function output = relu_func(z)
  output = 0;
  if (z > 0)
    output = z;
  else 
    output = 0;
  endif
endfunction
  
#Leaky ReLU Function
function output = leaky_relu_func(z, grad)
  output = 0;
  if (z > 0)
    output = z;
  else 
    output = grad * z;
  endif
endfunction