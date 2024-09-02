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
Ggm = zeros(Niter,1);

xprev = init; % initialize
for i=1:Niter
        xcurr = Prox(xprev); % update

        Xgm(:,i) = col(W(xcurr));
        Cgm(i) = F(xcurr);
	Ggm(i) = norm(Ld*(xcurr - xprev));

        xprev = xcurr;
end

% fgm
Xfgm = zeros(numel(xtrue), Niter);
Cfgm = zeros(Niter,1);
Gfgm = zeros(Niter,1);

yprev = init; xprev = init; % initialize
ti = 1;
for i=1:Niter
        xcurr = Prox(yprev); % update
        % momentum
        tip = (1 + sqrt(1 + 4*ti^2)) / 2;
        ycurr = xcurr + (ti - 1)/tip*(xcurr - xprev);

        Xfgm(:,i) = col(W(xcurr));
        Cfgm(i) = F(xcurr);
	Gfgm(i) = norm(Ld*(xcurr - yprev));

        xprev = xcurr;
        yprev = ycurr;
        ti = tip;
end

% fgmg
Xfgmg = zeros(numel(xtrue), Niter);
Cfgmg = zeros(Niter,1);
Gfgmg = zeros(Niter,1);

yprev = init; xprev = init; % initialize
ti = 1;
Ti = 1;
for i=1:Niter
        xcurr = Prox(yprev); % update
        % momentum
        if i <= ceil(Niter/2) - 1
		tip = (1 + sqrt(1 + 4*ti^2)) / 2;
        else
		tip = (Niter - i + 2) / 2;
	end
	Tip = Ti + tip;

	ycurr = xcurr + (Ti - ti)*tip/(ti*Tip) * (xcurr - xprev) ...
			+ (ti^2 - Ti)*tip/(ti*Tip) * (xcurr - yprev);

        Xfgmg(:,i) = col(W(xcurr));
        Cfgmg(i) = F(xcurr);
	Gfgmg(i) = norm(Ld*(xcurr - yprev)); 

        xprev = xcurr;
        yprev = ycurr;
        ti = tip;
	Ti = Tip;
end

% fgmg3
Xfgmg3 = zeros(numel(xtrue), Niter);
Cfgmg3 = zeros(Niter,1);
Gfgmg3 = zeros(Niter,1);

yprev = init; xprev = init; % initialize
ti = 1;
Ti = 1;
for i=1:Niter
        xcurr = Prox(yprev); % update
        % momentum
        tip = (i+3) / 3;
        Tip = Ti + tip;

        ycurr = xcurr + (Ti - ti)*tip/(ti*Tip) * (xcurr - xprev) ...
                        + (ti^2 - Ti)*tip/(ti*Tip) * (xcurr - yprev);

        Xfgmg3(:,i) = col(W(xcurr));
        Cfgmg3(i) = F(xcurr);
        Gfgmg3(i) = norm(Ld*(xcurr - yprev));

        xprev = xcurr;
        yprev = ycurr;
        ti = tip;
        Ti = Tip;
end


%%%%%%% display
%figure(1), im('notick', reshape(Xogm(:,end), size(xtrue)), ' ', [0 1]);
%print('-depsc', 'fig/deblur_l1_image.eps');

figure(2);
Iter = 1:Niter;
f2 = semilogy(Iter, Cgm - Fconv, 'k-.', ...
        Iter, Cfgm - Fconv, 'b--', ...
        Iter, Cfgmg - Fconv, 'r-', ...
	Iter, Cfgmg3 - Fconv, 'm-', ...
	Iter, repmat(Cfgm(end), [Niter 1]) - Fconv, 'k:');
axis([0 Niter min(Cfgm - Fconv) * 0.5 max(Cgm - Fconv) * 2]);
l2 = {'PGM', 'FPGM', 'FPGM-PG', 'FPGM-PG3'};
legend(f2, l2, 'fontsize', 25);
set(gca, 'fontsize', 22);
set(f2, 'markersize', 9, 'linewidth', 3);
%print('-depsc', 'fig/deblur_l1_plot.eps');

figure(3);
Iter = 1:Niter;
f3 = semilogy(Iter, Ggm, 'k-.', ...
        Iter, Gfgm, 'b--', ...
        Iter, Gfgmg, 'r-', ...
	Iter, Gfgmg3, 'm-', ...
        Iter, repmat(Gfgm(end), [Niter 1]), 'k:');
axis([0 Niter min(Gfgm) * 0.5 max(Ggm) * 2]);
l3 = {'PGM', 'FPGM', 'FPGM-PG', 'FPGM-PG3'};
legend(f3, l3, 'fontsize', 25);
set(gca, 'fontsize', 22);
set(f3, 'markersize', 9, 'linewidth', 3);


