% deblur_huber
% icip 2015 submission
% donghwan kim

clear all;
close all;

if 0
	load('mat/deblur_setup.mat');

	delta = 1e-2; %%% todo
	R = Reg1(mask, 'offsets', '2d:hvd', 'pot_arg', {'huber', delta});

	lam = 1e-3; %%% todo
	F = @(x) 1/2*norm(col(b - A*x))^2 + lam*R.penal(R, x(:));
	Fgrad = @(x) A'*(A*x - b) + lam*reshape(R.cgrad(R, x(:)), size(x)); 

	%% power iteration
	%p = xtrue(:);
	%for i=1:2000
        %        tmp = A'*(A*p);
        %        p = tmp / norm(col(tmp));
        %end
        %Ld = max(col((A'*(A*p)) ./ p));

	%p = xtrue(:);
	%C = R.C1;
	%wt = [R.wt.col(1); R.wt.col(2); R.wt.col(3); R.wt.col(4)];
	%for i=1:20000
        %        tmp = C'*(wt.*(C*p));
        %        p = tmp / norm(col(tmp));
        %end
        %Lr = max(col((C'*(wt.*(C*p))) ./ p));
	
	Ld = 1; % todo
	Lr = 8*1.7071; %%
	L = Ld + lam*Lr; 

	Prox = @(x) x - 1/L * Fgrad(x);

	% generate optimum
	yprev = xtrue; xprev = xtrue; % initialize
	ti = 1;
	for i=1:10000
		xcurr = Prox(yprev);
        	tip = (1 + sqrt(1 + 4*ti^2)) / 2;
        	ycurr = xcurr + (ti - 1)/tip*(xcurr - xprev);

        	xprev = xcurr;
        	yprev = ycurr;
        	ti = tip;
	end
	xconv = xcurr;
	Fconv = F(xcurr);

	figure(10), im(xtrue, [0 1]);
	figure(20), im(b, [0 1]);
	figure(30), im(xconv, [0 1]);

	save('mat/deblur_huber.mat'); 
	return;
else
	load('mat/deblur_huber.mat'); 
end

Niter = 200;

% gm
Xgm = zeros(numel(xtrue), Niter);
Cgm = zeros(Niter,1);

xprev = b; % initialize
for i=1:Niter
	xcurr = Prox(xprev); % update

	Xgm(:,i) = xcurr(:);
	Cgm(i) = F(xcurr);	

	xprev = xcurr;
end

% fgm
Xfgm = zeros(numel(xtrue), Niter);
Cfgm = zeros(Niter,1);

yprev = b; xprev = b; % initialize
ti = 1;
for i=1:Niter
        xcurr = Prox(yprev); % update
	% momentum
	tip = (1 + sqrt(1 + 4*ti^2)) / 2;
	ycurr = xcurr + (ti - 1)/tip*(xcurr - xprev); 

        Xfgm(:,i) = xcurr(:);
        Cfgm(i) = F(xcurr);       

        xprev = xcurr;
	yprev = ycurr;
	ti = tip;
end

% ogm
Xogm = zeros(numel(xtrue), Niter);
Cogm = zeros(Niter,1);

yprev = b; xprev = b; % initialize
ti = 1;
for i=1:Niter
        xcurr = Prox(yprev); % update
        % momentum
        tip = (1 + sqrt(1 + 4*ti^2)) / 2;
        ycurr = xcurr + (ti - 1)/tip*(xcurr - xprev) + ti/tip*(xcurr - yprev);

        Xogm(:,i) = xcurr(:);
        Cogm(i) = F(xcurr);         
        
        xprev = xcurr;
        yprev = ycurr;
        ti = tip;
end


%%%%%%% display
figure(1), im('notick', reshape(Xogm(:,end), size(xtrue)), ' ', [0 1]);
print('-depsc', 'fig/deblur_huber_image.eps');

figure(2);
Iter = 1:Niter;
f2 = semilogy(Iter, Cgm - Fconv, 'k-.', ...
	Iter, Cfgm - Fconv, 'b--', ...
	Iter, Cogm - Fconv, 'r-', ...
	Iter, repmat(Cfgm(end), [Niter 1]) - Fconv, 'k:');
axis([0 Niter min(Cogm - Fconv) * 0.5 max(Cgm - Fconv) * 2]);
l2 = {'GM', 'FGM', 'OGM'};
legend(f2, l2, 'fontsize', 25);
set(gca, 'fontsize', 22);
set(f2, 'markersize', 9, 'linewidth', 3);
print('-depsc', 'fig/deblur_huber_plot.eps');









