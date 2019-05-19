using Distributions: pdf, Normal, Uniform
using LinearAlgebra: diag
# Set initial setup of parameters
B = [4.0,0.7];
N = 100000;
K = size(B)[1];

# Construct x and y following normal linear model
s2 = 10;
d = Normal();
u = rand(Normal(0,sqrt(s2)),N);

x0 = ones(N);
x1 = rand(Uniform(),N);

X = [x0 x1];

y = X*B + u;

startv = [1.0,1.0,0.0];

function llf(param,y=y,X=X);
	Bhat = param[1:K];
	s2hat = exp(param[K+1]);
	xb = X*Bhat;
	ll = log.(pdf.(Normal(0,sqrt(s2hat)),y-xb));
	return(-sum(ll));
end

include("bfgsmin.jl")
llf(startv)
numgr(llf,startv)
hessian(llf,startv)
numhess(llf,startv)

@time llf(startv);
@time numgr(llf,startv; difftype="forward");

@time res = bfgsmin(llf,startv; difftype="forward",verbose=true,hess=true);
@time res2 = bfgsmin(llf,startv; difftype="forward",verbose=true,hess=false);

sqrt.(diag(inv(res["hessian"])))
sqrt.(diag(inv(res2["hessian"])))
