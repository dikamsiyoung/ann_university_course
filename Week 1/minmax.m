clear; 
x=4:0.005:5; 
y1=exp(x)+25*sin(100*x)+43*randn(1,201)+36*cos(10*x); 
min1=min(y1); 
max1=max(y1); 
min2=0; 
max2=1;
y2=((y1-min1)/(max1-min1))*(max2-min2) + min2; 
subplot(121), plot(y1(1:200)), xlabel('A') 
subplot(122), plot(y2(1:200)), xlabel('B')