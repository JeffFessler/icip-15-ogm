% deblur setup
% icip 2015 submission
% donghwan kim

clear all;
close all;

xtrue = double(imread('cameraman.tif'))' / 255;
h = fspecial('gaussian', [9 9], 4);
mask = true(size(xtrue));
A = Gblur(mask, 'psf', h, 'type', 'conv,per');
sigma = 1e-3;
b = A*xtrue + sigma*randn(size(xtrue));

figure(1), im('notick', xtrue, ' ', [0 1], '');
print('-depsc', 'fig/xtrue.eps');
figure(2), im('notick', b, ' ', [0 1], '');
print('-depsc', 'fig/xblurred.eps');

save('mat/deblur_setup.mat');


