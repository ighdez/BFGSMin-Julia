using LinearAlgebra

# Numerical gradient function
function gr(f,param,ep=1e-07);
	K = size(param)[1];
	
	gr = hcat(fill(NaN,K));
	
	for k in 1:K;
		ej = fill(0.0,K);
		ej[k] = 1;
		gr[k,1] = (f(param + ep*ej) - f(param - ep*ej))/(2*ep);
	end;
	
	return(gr);
end

# BFGS function
function bfgsmin(f,x0,maxiter=1000,tol=sqrt(eps()),verbose=false,hessian=false);
	
	# Initialize
	x = x0;
	f_val = f(x);
	f_old = copy(f_val);
	g0 = gr(f,x);
	H0 = Matrix{Float64}(I,size(x)[1],size(x)[1]);
	g_diff = Inf;
	c1 = 1e-04;
	lambda = 1.
	convergence = -1;
	iter = 0
	   
	# Start algorithm
	for it in 1:maxiter;
		println("Optimizing: Iter No: ",string(iter)," / F-Value: ", string(round(f_val;digits=2))," / Step: ",string(round(lambda;digits=5))," / G-Diff: ",string(round(g_diff;digits=6)))
		lambda = 1;
		
		if g_diff <= (1 + abs(f_old))*tol*tol;
			println("Converged!")
			convergence = 0;
			break
		end
		
		# Construct direction vector
		d = inv(-H0)*g0;
		m = d'*g0;
		
		# Select step to satisfy the Armijo's Rule
		while true;
			x1 = x + lambda*d[:,1];
			f1 = try 
					f(x1)
				catch
					NaN
				end

			ftest = f_val + c1*lambda*m[1];

			if isfinite(f1) & (f1 <= ftest);
				break
			else
				lambda = lambda*0.2;
			end
		end

		# Construct the improvement and relative gradient improvement
		x1 = x + lambda*d[:,1];
		s0 = lambda*d;
		
		g1 = gr(f,x1);
		
		y0 = g1 - g0;
		
		H1 = H0 + (y0*y0')./(y0'*s0) - (H0*s0*s0'*H0)./(s0'*H0*s0);
		g0 = copy(g1);
		f_new = f(x1);
		g_diff = abs(m[1]);
		f_old = copy(f_val);
		f_val = copy(f_new);
		x = copy(x1);
		H0 = copy(H1);
		iter = iter + 1
	end
	
	if iter == maxiter;
		convergence = 2;
	end
	
	h_final = H0;
	
	results = Dict("convergence" => convergence, "iterations" => iter, "max_f" => f_val, "par_max" => x, "hessian" => h_final);
	
	return(results);
end