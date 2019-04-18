#################################################
#												
# 			   BFGSMin - Julia v0.3			
#												
# Changelog:									
#												
#														
# v0.3: - 	Inclusion of improved hessian that
# 			uses broadcast capabilities of Julia,
#			based on the function 'Numerical_Hessian'
#			included in the R package 'CDM'.
#
#		-	Now its possible to add a user-written
#			gradient function.
#
#		-	The non-improved gradient function was
#			deprecated.
#
#		-	A lot of bug fixes.
#
# v0.2: Inclusion of improved gradient for 		
#       optimization with high amount of data 	
#       or complicated LL Functions.
#
# v0.1: Initial version, at least operational.			
#												
#################################################

using LinearAlgebra: inv, Matrix, I, norm, diagind
using Calculus: hessian

# Improved numerical gradient function (with broadcast operations)
function numgr(f,param; difftype="forward",ep=sqrt(eps()));
	K = size(param)[1];
	gr = fill(NaN,K)
	plus = fill(Float64[],K,1)
	minus = fill(Float64[],K,1)
	ej = Matrix{Float64}(I,K,K)*ep;
	
	# Generate step matrices
	for i = 1:K
		plus[i] = param .+ ej[i,:]
		minus[i] = param .- ej[i,:]
	end
	
	# Generate central or forward gradient
	if difftype == "central"
		gr = @. (f(plus) - f(minus))*0.5/ep
	elseif difftype == "forward"
		gr = (f.(plus) .- f(param))./ep
	else 
		error("Non-valid difference type in gradient")
	end
	return(gr);
end

# Improved numerical hessian function (with broadcast operations)
function numhess(f,param; ep=1e-05)
	# This code is based in the R function'Numerical_Hessian'
	# created by Alexander Robitzsch for the package 'CDM'
	# I acknowledge him and his team for their efforts.

	# Initialize
	K = size(param)[1];
	hs = fill(NaN,K,K);
	plus = fill(Float64[],K,1);
	plus2 = fill(Float64[],K,1);
	plusplus = fill(Float64[],K,K);
	ej = Matrix{Float64}(I,K,K)*ep;
	
	# Generate step matrices
	for i = 1:K;
		plus[i] = param .+ ej[i,:];
		plus2[i] = param .+ 2*ej[i,:];
		
		for j = i+1:K;
				plusplus[i,j] = param .+ ej[i,:] .+ ej[j,:];
				plusplus[j,i] = plusplus[i,j];
		end;
		plusplus[i,i] = zeros(K);
	end;
	
	# Generate relevant evaluations
	fplus = f.(plus);
	fplus2 = f.(plus2);
	fpp = f.(plusplus);
	fx = f(param);
	
	# Compute hessian by broadcasting
	hs = (fpp .- fplus  .- fplus' .+ fx)./(ep*ep);
	hs[diagind(hs)] = (fplus2 - 2*fplus .+ fx)./(ep*ep);

	return(hs)
end

# BFGS function
function bfgsmin(f,x0,g=nothing; maxiter=1000,tol=1e-06,verbose=false,hess=false,difftype="central", diffeps=sqrt(eps()));
	
	# Initialize
	x = x0;
	f_val = f(x);
	f_old = copy(f_val);
	if isnothing(g)
		g0 = numgr(f,x; difftype=difftype,ep=diffeps);
	else
		g0 = g(x)
	end

	H0 = Matrix{Float64}(I,size(x)[1],size(x)[1]);
	g_diff = Inf;
	c1 = 1e-04;
	lambda = 1.
	convergence = -1;
	iter = 0
	
	if verbose
		if isnothing(g)
			println("\n","Warning: BFGSMin will use a finite-difference gradient")
		end
		
		if !hess
			println("\n","Warning: BFGSMin will return the approximate BFGS Hessian update")
		end
		
		println("\n","Optimizing: Initial F-Value: ", string(round(f_val;digits=2)))
	end
	# Start algorithm
	for it = 1:maxiter;
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
				lambda = lambda./2;
			end
		end

		# Construct the improvement and gradient improvement
		x = x + lambda*d[:,1];
		
		# Update Hessian
		
		if isnothing(g)
			g1 = numgr(f,x; difftype=difftype,ep=diffeps);
		else
			g1 = g(x);
		end
		
		s0 = lambda*d;
		y0 = g1 - g0;
		
		H0 = H0 + (y0*y0')./(y0'*s0) - (H0*s0*s0'*H0)./(s0'*H0*s0);
		
		g0 = copy(g1);
		f_val = f(x);
		
		g_diff = abs(m[1]);
		
		# Check if relative gradient is less than tolerance
		if g_diff < tol;
			if verbose
				println("\n","Converged!")
			end
			convergence = 0;
			break
		end
		
		iter = iter + 1
		
		# Show information if verbose == true
		if verbose
			println("Optimizing: Iter No: ",string(iter)," / F-Value: ", string(round(f_val;digits=2))," / |g(x)'(-H(x)^-1)g(x)|: ",string(round(g_diff;digits=6))," / Step: ",string(round(lambda;digits=5)))
		end

	end
	
	if iter == maxiter;
		convergence = 2;
		println("\n","Maximum iterations reached reached. Convergence not achieved.")
	end
	
	if hess
		if verbose
			println("\n","Computing approximate Hessian")
		end
		
		h_final = numhess(f,x);
	else
		h_final = H0;
	end
	
	results = Dict("convergence" => convergence, "iterations" => iter, "max_f" => f_val, "par_max" => x, "hessian" => h_final);
	
	return(results);
end