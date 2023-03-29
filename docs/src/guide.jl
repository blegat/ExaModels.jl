# # Getting Started
# SIMDiff provides a built-in API for creating nonlinear prgogramming models and allows solving the created models using NLP solvers (in particular, those that are interfaced with `NLPModels`, such as [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl). We now use `SIMDiff`'s bulit-in API to model the following nonlinear program:
# ```math
# \begin{aligned}
# \min_{\{x_i\}_{i=0}^N} &\sum_{i=2}^N  100(x_{i-1}^2-x_i)^2+(x_{i-1}-1)^2\\
# \text{s.t.} &  3x_{i+1}^3+2x_{i+2}-5+\sin(x_{i+1}-x_{i+2})\sin(x_{i+1}+x_{i+2})+4x_{i+1}-x_i e^{x_i-x_{i+1}}-3 = 0
# \end{aligned}
# ```
# We model the problem with:
N = 10000

# First, we create a `SIMDiffModel`.
m = SIMDiffModel() 

# The variables can be created as follows:
x = [variable(m; start = mod(i,2)==1 ? -1.2 : 1.) for i=1:N];

# The objective can be set as follows:
objective(m, sum(100(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i=2:N));

# The constraints can be set as follows:
for i=1:N-2
    constraint(m, 3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3 == 0);
end

# The important last step is instantiating the model. This step must be taken before calling optimizers.
instantiate!(m)

# To solve the problem with `Ipopt`,
using NLPModelsIpopt
sol = ipopt(m);

# The solution `sol` contains the field `sol.solution` holding the optimized parameters.

# ### SIMDiff as an AD backend of JuMP
# SIMDiff can be used as an automatic differentiation backend of JuMP. The problem above can be modeled in `JuMP` and solved with `Ipopt` along with `SIMDiff`

using JuMP, Ipopt

m = JuMP.Model(Ipopt.Optimizer) 

@variable(m, x[i=1:N], start=mod(i,2)==1 ? -1.2 : 1.)
@NLobjective(m, Min, sum(100(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i=2:N))
@NLconstraint(m, [i=1:N-2], 3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3 == 0)

optimize!(m; differentiation_backend = SIMDiffAD())
