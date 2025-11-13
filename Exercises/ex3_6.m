clc,clear,close all

addpath('../Code/functions', '../Code/data') 

A=[1 -1.5 0.8];
N=1000;
extraN=100;
e=randn(N+extraN,1);
y=filter(1,A,e); y=y(extraN+1:end);
noLags = 20;
figure; plotACFnPACF( y, noLags, 'AR');
figure; zplane(A)

Padd = 1024*4;
Y = fftshift( abs( fft(y.*hamming(N), Padd) ).^2 / N );
figure; plot(Y)