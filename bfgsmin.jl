#################################################
#												
# 			   BFGSMin - Julia v0.2			
#												
# Changelog:									
#												
#														
# v0.1: Initial version, at least operational.	
#												
# v0.2: Inclusion of improved gradient for 		
#       optimization with high amount of data 	
#       or complicated LL Functions.			
#												
#################################################

using LinearAlgebra: inv, Matrix, I, norm
using Calculus: hessian

# Numerical gradient function
function gr(f,param; difftype="central",ep=sqrt(eps()));
	K = size(param)[1];
	
	gr = fill(NaN,K,1);
	ej = Matrix{Float64}(I,K,K);
	for k = 1:K;
		if difftype == "central"
			gr[k,1] = (f(param + ep*ej[:,k]) - f(param - ep*ej[:,k]))./(2*ep);
		elseif difftype == "forward"
			gr[k,1] = (f(param + ep*ej[:,k]) - f(param))./ep;
		else 
			error("Non-valid difference type in gradient")
		end
	end
	
	return(gr);
end

# Improved numerical gradient function (with vectorized operations)
function gr2(f,param; difftype="central",ep=sqrt(eps()));
	K = size(param)[1];
	
	plus = fill(Float64[],K,1)
	minus = fill(Float64[],K,1)

	ej = Matrix{Float64}(I,K,K)*ep;
	
	for i = 1:K
		plus[i] = param .+ ej[i,:]
		minus[i] = param .- ej[i,:]
	end
	if difftype == "central"
		gr = (f.(plus) .- f.(minus))./(2*ep)
	elseif difftype == "forward"
		gr = (f.(plus) .- f(param))./ep
	else 
		error("Non-valid difference type in gradient")
	end
	return(gr);
end

# BFGS function
function bfgsmin(f,x0; maxiter=1000,tol=1e-06,verbose=false,hess=false,difftype="central", diffeps=sqrt(eps()),gr=gr);
	
	# Initialize
	x = x0;
	f_val = f(x);
	f_old = copy(f_val);
	g0 = gr(f,x; difftype=difftype,ep=diffeps);
	H0 = Matrix{Float64}(I,size(x)[1],size(x)[1]);
	g_diff = Inf;
	c1 = 1e-04;
	lambda = 1.
	convergence = -1;
	iter = 0
	   
	# Start algorithm
	for it = 1:maxiter;
		println("Optimizing: Iter No: ",string(iter)," / F-Value: ", string(round(f_val;digits=2))," / | g(x)'(-H(x)^-1)g(x) |: ",string(round(g_diff;digits=6))," / Step: ",string(round(lambda;digits=5)))
		lambda = 1;

		# Construct direction vector and relative gradient
		d = inv(-H0)*g0;
		m = d'*g0;
		
		# Select step to satisfy the Armijo-Goldstein condition
		while true;
			x1 = x + lambda*d[:,1];
			f1 = try 
					f(x1)
				catch
					NaN
				end

			ftest = f_val + c1*lambda*m[1];

			if isfinite(f1) & (f1 <= ftest) & (f1>0);
				break
			else
				lambda = lambda*0.2;
			end
		end

		# Construct the improvement and gradient improvement
		x = x + lambda*d[:,1];
		
		# Update Hessian
		s0 = lambda*d;
		g1 = gr(f,x; difftype=difftype,ep=diffeps);
		y0 = g1 - g0;
		
		H0 = H0 + (y0*y0')./(y0'*s0) - (H0*s0*s0'*H0)./(s0'*H0*s0);
		
		g0 = copy(g1);
		f_val = f(x);
		g_diff = abs(m[1]);
		
		# Check if relative gradient is less than tolerance
		if g_diff <= (1+abs(f_val))*tol*tol;
			println("Converged!")
			convergence = 0;
			break
		end
		
		iter = iter + 1
	end
	
	if iter == maxiter;
		convergence = 2;
	end
	
	if hess
		println("Computing approximate Hessian")
		h_final = hessian(llf,x);
	else
		println("Warning: BFGSMin will return the approximate BFGS Hessian update")
		h_final = H0;
	end
	
	results = Dict("convergence" => convergence, "iterations" => iter, "max_f" => f_val, "par_max" => x, "hessian" => h_final);
	
	return(results);
end