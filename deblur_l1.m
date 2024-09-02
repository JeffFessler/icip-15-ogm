% deblur_l1
% icip 2015 submission
% donghwan kim

clear all;
close all;

if 0
	load('mat/deblur_setup.mat');
	% the inverse of a three stage Haar wavelet transform
	[C, S] = wavedec2(xtrue, 3, 'haar');

	W = @(x) waverec2(x,S,'haar');
	tW = @(x) reshape(wavedec2(x,3,'haar'), size(x));
	
	fgrad = @(x) tW(A'*(A*W(x) - b));

	lam = 2e-5; 
        F = @(x) 1/2*norm(col(b - A*W(x)))^2 + lam*norm(x(:),1);

	%% power iteration
        p = randn(size(xtrue));
        for i=1:2000
               tmp = tW(A'*A*W(p));
               p = tmp / norm(col(tmp));
        end
        Ld = max(col(tW(A'*A*W(p)) ./ p))
	%Ld = 1; % todo

	shrink = @(g, alp) (abs(g) - alp).* (abs(g) >= alp) .* sign(g);
        Prox = @(x) shrink(x - 1/Ld*fgrad(x), lam/Ld); 

        % generate optimum
        init = tW(xtrue);
	yprev = init; xprev = init; % initialize
        ti = 1;
        for i=1:10000
                xcurr = Prox(yprev);
                tip = (1 + sqrt(1 + 4*ti^2)) / 2;
                ycurr = xcurr + (ti - 1)/tip*(xcurr - xprev);

                xprev = xcurr;
                yprev = ycurr;
                ti = tip;
        end
        xconv = W(xcurr);
        Fconv = F(xcurr);

        figure(10), im(xtrue, [0 1]);
        figure(20), im(b, [0 1]);
        figure(30), im(xconv, [0 1]);

        save('mat/deblur_l1.mat');
        return;
else
        load('mat/deblur_l1.mat');
end

Niter = 200;
init = tW(b);

% gm
Xgm = zeros(numel(xtrue), Niter);
Cgm = zeros(Niter,1);

xprev = init; % initialize
for i=1:Niter
        xcurr = Prox(xprev); % update

        Xgm(:,i) = col(W(xcurr));
        Cgm(i) = F(xcurr);

        xprev = xcurr;
end

% fgm
Xfgm = zeros(numel(xtrue), Niter);
Cfgm = zeros(Niter,1);

yprev = init; xprev = init; % initialize
ti = 1;
for i=1:Niter
        xcurr = Prox(yprev); % update
        % momentum
        tip = (1 + sqrt(1 + 4*ti^2)) / 2;
        ycurr = xcurr + (ti - 1)/tip*(xcurr - xprev);

        Xfgm(:,i) = col(W(xcurr));
        Cfgm(i) = F(xcurr);

        xprev = xcurr;
        yprev = ycurr;
        ti = tip;
end

% ogm
Xogm = zeros(numel(xtrue), Niter);
Cogm = zeros(Niter,1);

yprev = init; xprev = init; % initialize
ti = 1;
for i=1:Niter
        xcurr = Prox(yprev); % update
        % momentum
        tip = (1 + sqrt(1 + 4*ti^2)) / 2;
        ycurr = xcurr + (ti - 1)/tip*(xcurr - xprev) + ti/tip*(xcurr - yprev);

        Xogm(:,i) = col(W(xcurr));
        Cogm(i) = F(xcurr);

        xprev = xcurr;
        yprev = ycurr;
        ti = tip;
end

%%%%%%% display
figure(1), im('notick', reshape(Xogm(:,end), size(xtrue)), ' ', [0 1]);
print('-depsc', 'fig/deblur_l1_image.eps');

figure(2);
Iter = 1:Niter;
f2 = semilogy(Iter, Cgm - Fconv, 'k-.', ...
        Iter, Cfgm - Fconv, 'b--', ...
        Iter, Cogm - Fconv, 'r-', ...
	Iter, repmat(Cfgm(end), [Niter 1]) - Fconv, 'k:');
axis([0 Niter min(Cogm - Fconv) * 0.5 max(Cgm - Fconv) * 2]);
l2 = {'ISTA', 'FISTA', 'OISTA'};
legend(f2, l2, 'fontsize', 25);
set(gca, 'fontsize', 22);
set(f2, 'markersize', 9, 'linewidth', 3);
print('-depsc', 'fig/deblur_l1_plot.eps');


