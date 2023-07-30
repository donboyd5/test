# https://jso.dev/
# https://jso.dev/tutorials/introduction-to-nlpmodelsipopt/
# https://jso.dev/tutorials/solve-an-optimization-problem-with-ipopt/

using Distributions
using DataFrames, LinearAlgebra, NLPModels, NLPModelsIpopt, Random
using ADNLPModels
using SparseArrays


mutable struct ReweightProblem
    wh::Vector{Float64}
    xmat::Matrix{Float64}
    rwtargets::Vector{Float64}
    h::Int  # number of households
    k::Int # number of characteristics per household
    rwtargets_calc::Vector{Float64}
    rwtargets_diff::Vector{Float64}
  
    # placeholders in case we create scaled versions of the data
    wh_scaled::Vector{Float64}
    xmat_scaled::Matrix{Float64}
    rwtargets_scaled::Vector{Float64}
  
    function ReweightProblem(wh, xmat, rwtargets)
        # check dimensions
        length(wh) == size(xmat, 1) || throw(DimensionMismatch("wh and xmat must have same # of rows"))
        size(xmat, 2) == length(rwtargets) || throw(DimensionMismatch("xmat # of columns must equal length of reweight_targets"))
  
        rwtargets_calc = xmat' * wh
        rwtargets_diff = rwtargets_calc - rwtargets
  
        h = length(wh)
        k = size(xmat)[2]
        new(wh, xmat, rwtargets, h, k, rwtargets_calc, rwtargets_diff)
    end
  end

function mtprw(h, k; pctzero=0.0)
    Random.seed!(123)
    xsd=.005
    wsd=.005
    # pctzero=0.10
    # h = 8
    # k = 2

    # create xmat
    d = Normal(0., xsd)
    r = rand(d, (h, k)) # xmat dimensions
    xmat = 100 .+ 20 .* r

    # set random indexes to zero
    nzeros = round(Int, pctzero * length(xmat))
    # Generate a random set of indices to set to zero
    zindices = randperm(length(xmat))[1:nzeros]
    # Set the elements at the selected indices to zero
    xmat[zindices] .= 0.0 # this works even when pctzero is 0.0

    # create wh
    d = Normal(0., wsd)
    r = rand(d, h) # wh dimensions
    wh = 10 .+ 10 .* (1 .+ r)

    # calc  sums and add noise to get targets
    rwtargets = xmat' * wh
    r = rand(Normal(0., 0.1), k)
    rwtargets = rwtargets .* (1 .+ r)

    return ReweightProblem(wh, xmat, rwtargets)
end


tp = mtprw(100, 4, pctzero=.3)
tp = mtprw(1000, 10, pctzero=.3)
tp = mtprw(10_000, 40, pctzero=.3)

tp.h
tp.wh
tp.xmat
tp.rwtargets

rwtargets_calc = tp.xmat' * tp.wh
ratio0 = ones(tp.h)
rwtargets_calc = tp.xmat' * (ratio0 .* tp.wh)

f(ratio) = sum((ratio .- 1.).^2)
f(ratio0)

c(ratio) = tp.xmat' * (ratio .* tp.wh)
c(ones(tp.h))

lvar = fill(0.1, tp.h)
uvar = fill(2.0, tp.h)

tol = .01
lcon = tp.rwtargets .* (1 - tol)
ucon = tp.rwtargets .* (1 + tol)

ratio0 = ones(tp.h)
# ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon) 

c(ratio0) ./ tp.rwtargets .- 1

bjac = sparse(tp.xmat .!= 0)
bjac = sparse(tp.xmat)
fieldnames(typeof(bjac))
bjac.m
bjac.n
bjac.colptr
bjac.rowval
bjac.nzval


nlp = ADNLPModel(
    f,
    ratio0,
    lvar,
    uvar,
    c,
    lcon,
    ucon
)

# MA27
# MUMPS, MA57, MA86
# hsllib = "/usr/local/lib/libcoinhsl.so"
# hsllib = "/usr/local/lib/libhsl.so"
# hsllib = "/usr/local/lib/libhsl.so"
# hsllib = "/home/donboyd5/Documents/julia_projects/test/libhsl.so"
hsllib = "/usr/local/lib/lib/x86_64-linux-gnu/libhsl.so"

# ma27: use the Harwell routine MA27
# ma57: use the Harwell routine MA57
# ma77: use the Harwell routine HSL_MA77
# ma86: use the Harwell routine HSL_MA86
# ma97: use the Harwell routine HSL_MA97

# res = ipopt(nlp, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma57")
res = ipopt(nlp, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma77")

pwd()
rm("ma77_delay", force=true)
rm("ma77_int", force=true)
rm("ma77_real", force=true)
rm("ma77_work", force=true)

fieldnames(typeof(res))
res.objective
res.solution
quantile(res.solution)

c(res.solution)
tp.rwtargets

c(ratio0) ./ tp.rwtargets .- 1
c(res.solution) ./ tp.rwtargets .- 1

tp.xmat

bjac = sparse(tp.xmat .!= 0)

nlp = ADNLPModel(
  x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2), # f(x)
  [0.5; 0.5], # starting point, which can be your guess
  [-Inf; 0.25], # lower bounds on variables
  [0.5; 0.75],  # upper bounds on variables
  x -> [x[1]^2 + x[2]^2], # constraints function - must be an array
  [-Inf], # lower bounds on constraints
  [1.0]   # upper bounds on constraints
)

output = ipopt(nlp, print_level=5)


# another example
using Ipopt, JuMP
optimizer = with_optimizer(Ipopt.Optimizer, linear_solver = "ma27")
optimizer = solver(Ipopt.Optimizer, linear_solver = "ma27")
model = Model(optimizer)
@variable(model, 0 <= x <= 2)
@variable(model, 0 <= y <= 30)
@objective(model, Max, 5x + 3 * y)
@constraint(model, con, 1x + 5y <= 3)
optimize!(model)

# build an nlp model from scratch =============================================================================================
# https://jso.dev/tutorials/introduction-to-nlpmodelsipopt/
# https://github.com/JuliaSmoothOptimizers/NLPModelsTest.jl
tp = mtprw(10, 2, pctzero=.3)
tp = mtprw(100, 4, pctzero=.3)

ratio0 = ones(tp.h)
rwtargets_calc = tp.xmat' * (ratio0 .* tp.wh)

f(ratio) = sum((ratio .- 1.).^2)
f(ratio0)

c(ratio) = tp.xmat' * (ratio .* tp.wh)
c(ones(tp.h))

lvar = fill(0.1, tp.h)
uvar = fill(2.0, tp.h)

tol = .01
lcon = tp.rwtargets .* (1 - tol)
ucon = tp.rwtargets .* (1 + tol)

# ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon) 

c(ratio0) ./ tp.rwtargets .- 1

bjac = sparse(tp.xmat .!= 0)
bjac = sparse(tp.xmat)

function NLPModels.obj(nlp, x)
    return sum((x .- 1.).^2)
end

function NLPModels.grad!(nlp, x, g)
    g .= 2 .* (ratio .- 2.0)
end

function NLPModels.hess_structure!(nlp :: LogisticRegression, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
    n = nlp.meta.nvar
    I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
    rows[1 : nlp.meta.nnzh] .= [getindex.(I, 1); 1:n]
    cols[1 : nlp.meta.nnzh] .= [getindex.(I, 2); 1:n]
    return rows, cols
end


ratio = randn(10)
g .= 2 .* (ratio .- 2.0)



function NLPModels.grad!(nlp :: LogisticRegression, β::AbstractVector, g::AbstractVector)
    hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
    g .= nlp.X' * (hβ .- nlp.y) + nlp.λ * β
  end


  # info below here -------------------------------------------------------------------------------------------
  using DataFrames, LinearAlgebra, NLPModels, NLPModelsIpopt, Random

  mutable struct LogisticRegression <: AbstractNLPModel{Float64, Vector{Float64}}
    X :: Matrix{Float64}
    y :: Vector{Float64}
    λ :: Float64
    meta :: NLPModelMeta{Float64, Vector{Float64}} # required by AbstractNLPModel
    counters :: Counters # required by AbstractNLPModel
  end
  


  function LogisticRegression(X, y, λ = 0.0)
    m, n = size(X)
    meta = NLPModelMeta(n, name="LogisticRegression", nnzh=div(n * (n+1), 2) + n) # nnzh is the length of the coordinates vectors
    return LogisticRegression(X, y, λ, meta, Counters())
  end
  
  function NLPModels.obj(nlp :: LogisticRegression, β::AbstractVector)
    hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
    return -sum(nlp.y .* log.(hβ .+ 1e-8) .+ (1 .- nlp.y) .* log.(1 .- hβ .+ 1e-8)) + nlp.λ * dot(β, β) / 2
  end
  
  function NLPModels.grad!(nlp :: LogisticRegression, β::AbstractVector, g::AbstractVector)
    hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
    g .= nlp.X' * (hβ .- nlp.y) + nlp.λ * β
  end
  
  function NLPModels.hess_structure!(nlp :: LogisticRegression, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
    n = nlp.meta.nvar
    I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
    rows[1 : nlp.meta.nnzh] .= [getindex.(I, 1); 1:n]
    cols[1 : nlp.meta.nnzh] .= [getindex.(I, 2); 1:n]
    return rows, cols
  end
  
  function NLPModels.hess_coord!(nlp :: LogisticRegression, β::AbstractVector, vals::AbstractVector; obj_weight=1.0, y=Float64[])
    n, m = nlp.meta.nvar, length(nlp.y)
    hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
    fill!(vals, 0.0)
    for k = 1:m
      hk = hβ[k]
      p = 1
      for j = 1:n, i = j:n
        vals[p] += obj_weight * hk * (1 - hk) * nlp.X[k,i] * nlp.X[k,j]
        p += 1
      end
    end
    vals[nlp.meta.nnzh+1:end] .= nlp.λ * obj_weight
    return vals
  end
  
  Random.seed!(0)
  
  # Training set
  m = 1000
  df = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)
  df.buy = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)
  
  X = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]
  y = df.buy
  
  λ = 1.0e-2
  nlp = LogisticRegression(X, y, λ)
  stats = ipopt(nlp, print_level=0)
  β = stats.solution
  
  # Test set - same generation method
  m = 100
  df = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)
  df.buy = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)
  
  X = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]
  hβ = 1 ./ (1 .+ exp.(-X * β))
  ypred = hβ .> 0.5
  
  acc = count(df.buy .== ypred) / m
  println("acc = $acc")

# working example
using NLPModels
using LinearAlgebra
using SparseArrays

  # Define the model
mutable struct MyModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
end
  
  # Objective function
  function NLPModels.obj(nlp :: MyModel, x :: AbstractVector)
    return sum((1 .- x).^2)
  end
  
  # Gradient of the objective function
  function NLPModels.grad!(nlp :: MyModel, x :: AbstractVector, g :: AbstractVector)
    g .= -2 .* (1 .- x)
    # return g
  end
  
  # Hessian of the objective function
  function NLPModels.hess(nlp :: MyModel, x :: AbstractVector)
    n = nlp.meta.nvar
    H = Matrix{Float64}(I, n, n) * 2
    return H
  end


# SPARSE Hessian of the objective function
function NLPModels.hess(nlp :: MyModel, x :: AbstractVector)
  n = nlp.meta.nvar
  H = spdiagm(0 => fill(2.0, n))  # Creates a sparse diagonal matrix with all elements equal to 2.0
  return H
end

  
  # Number of variables
  n = 100_000
  
  # Starting point
  x0 = zeros(n)
  
  # Create an instance of MyModel
  nlp = MyModel(NLPModelMeta(n, x0=x0), Counters())
  
# Preallocate g
g = zeros(n)

# Precompute H
H = hess(nlp, x0)

# Minimize using Newton's method
for i = 1:10
    grad!(nlp, x0, g)
    d = -H \ g
    x0 += d
  end

x0  



# Minimize using Newton's method
# for i = 1:10
#     grad!(nlp, x0, g)
#     H = hess(nlp, x0)
#     d = -H \ g
#     x0 += d
# end


# more-complete solution
using NLPModels, NLPModelsIpopt, SparseArrays

# Define the model
mutable struct MyModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
end
  

# Objective function
function NLPModels.obj(nlp :: MyModel, x :: AbstractVector)
  return sum((1 .- x).^2)
end

# Gradient of the objective function
function NLPModels.grad!(nlp :: MyModel, x :: AbstractVector, g :: AbstractVector)
  g .= -2 .* (1 .- x)
  return g
end

# Hessian of the objective function
function NLPModels.hess(nlp :: MyModel, x :: AbstractVector)
  n = nlp.meta.nvar
  H = spdiagm(0 => fill(2.0, n))
  return H
end

# Hessian structure
function NLPModels.hess_structure!(nlp :: MyModel, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
    rows = 1:nlp.meta.nvar
    cols = 1:nlp.meta.nvar
    return rows, cols
  end

  function NLPModels.hess_coord!(
    nlp::MyModel,
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
  ) where {T}
    vals = fill(2.0, nlp.meta.nvar)
    return vals
  end




  # another try -------
  mutable struct HS6{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
  end
  
  function HS6(::Type{T}) where {T}
    meta = NLPModelMeta{T, Vector{T}}(
      2,
      ncon = 1,
      nnzh = 1,
      nnzj = 2,
      x0 = T[-1.2; 1],
      lcon = T[0],
      ucon = T[0],
      name = "HS6_manual",
    )
  
    return HS6(meta, Counters())
  end
HS6() = HS6(Float64)
fieldnames(typeof(HS6))
fieldnames(typeof(HS6()))
HS6().meta
HS6().meta.x0
HS6().meta.lcon

HS6() = HS6(10.0)
  

  # Hessian values
  function NLPModels.hess_coord!(nlp :: MyModel, x :: AbstractVector, vals :: AbstractVector)
    vals = fill(2.0, nlp.meta.nvar)
    return vals
  end



  function NLPModels.hess_coord!(
    nlp::HS6,
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
  ) where {T}
    @lencheck 2 x
    @lencheck 1 vals
    increment!(nlp, :neval_hess)
    vals[1] = 2obj_weight
    return vals
  end


# Number of variables
n = 5

# Starting point
x0 = zeros(n)

# Lower and upper bounds
lvar = zeros(n)
uvar = fill(2.0, n)

# Create an instance of MyModel
nlp = MyModel(NLPModelMeta(n, x0=x0, lvar=lvar, uvar=uvar), Counters())

res = ipopt(nlp, print_level=5, hessian_constant="yes")

res = ipopt(nlp, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma77")

# Solve the problem using Ipopt
solver = IpoptSolver(print_level=0)  # adjust print_level to control the output
result = solve(solver, nlp, x0)

# Print the solution
println("Solution x = ", result.solution)


mutable struct mymod{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
end

function mymod(::Type{T}) where {T}
    meta = NLPModelMeta{T, Vector{T}}(
      nvar;
      nnzh=nvar,
      x0 = zeros(T, nvar),
      lvar,
      uvar,
      name = "test"
    )
    return mymod(meta, Counters())
end
mymod() = mymod(5, lvar=fill(0., 5), uvar=fill(2., 5))
fieldnames(typeof(mymod))
fieldnames(typeof(mymod()))
mymod.var
mymod.meta.nvar
      

# NLPModelMeta(n, x0=x0, lvar=lvar, uvar=uvar),
  
#   function LINCON(::Type{T}) where {T}
#     meta = NLPModelMeta{T, Vector{T}}(
#       15,
#       nnzh = 15,
#       nnzj = 17,
#       ncon = 11,
#       x0 = zeros(T, 15),
#       lcon = T[22; 1; -Inf; -11; -1; 1; -5; -6; -Inf * ones(3)],
#       ucon = T[22; Inf; 16; 9; -1; 1; Inf * ones(2); 1; 2; 3],
#       name = "LINCON_manual",
#       lin = 1:11,
#       lin_nnzj = 17,
#       nln_nnzj = 0,
#     )


using NLPModels

function objective(x)
  return sum((1. .- x).^2)
end

function gradient(x)
  return (-2 * (1. .- x))
end

function hessian(x)
  return (2 * diag(ones(length(x))))
end

model = NLPModels.Problem(objective, gradient, hessian, x0 = [0.5, 0.5])

solver = NLPModels.IpoptSolver()

result = solver(model)

println(result)
  
# another try
import NLPModels: increment!
using NLPModels

mutable struct HS6 <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function HS6()
  meta = NLPModelMeta(2, ncon=1, nnzh=1, nnzj=2, x0=[-1.2; 1.0], lcon=[0.0], ucon=[0.0], name="hs6")

  return HS6(meta, Counters())
end

function NLPModels.obj(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return (1 - x[1])^2
end

function NLPModels.grad!(nlp :: HS6, x :: AbstractVector, gx :: AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= [2 * (x[1] - 1); 0.0]
  return gx
end

function NLPModels.hess(nlp :: HS6, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return [2.0 * obj_weight - 20 * w   0.0; 0.0 0.0]
end

function NLPModels.hess_coord(nlp :: HS6, x :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hess)
  w = length(y) > 0 ? y[1] : 0.0
  return ([1], [1], [2.0 * obj_weight - 20 * w])
end

function NLPModels.hprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
  increment!(nlp, :neval_hprod)
  w = length(y) > 0 ? y[1] : 0.0
  Hv .= [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
  return Hv
end

function NLPModels.cons!(nlp :: HS6, x :: AbstractVector, cx :: AbstractVector)
  increment!(nlp, :neval_cons)
  cx[1] = 10 * (x[2] - x[1]^2)
  return cx
end

function NLPModels.jac(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return [-20 * x[1]  10.0]
end

function NLPModels.jac_coord(nlp :: HS6, x :: AbstractVector)
  increment!(nlp, :neval_jac)
  return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end

function NLPModels.jprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod!(nlp :: HS6, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= [-20 * x[1]; 10] * v[1]
  return Jtv
end


# try with Standalone
# https://git.sr.ht/~cgeoga/StandaloneIpopt.jl/tree/master/item/src/ipopt_optimize.jl
using StandaloneIpopt

# ipopt_optimize(obj, ini, constraints=noconstraints(); 
#                         derivs=AutoFwdDerivs(obj, length(ini)),
#                         conderivs=AutoFwdConDerivs(length(ini), nconstr(constraints)),
#                         jac_nz=length(ini)*nconstr(constraints),
#                         box_lower=-1e22, box_upper=1e22, kwargs...)

ipopt_optimize(obj, ini, constraints=noconstraints(); 
                        derivs=AutoFwdDerivs(obj, length(ini)),
                        conderivs=AutoFwdConDerivs(length(ini), nconstr(constraints)),
                        jac_nz=length(ini)*nconstr(constraints),
                        box_lower=-1e22, box_upper=1e22, kwargs...)


using JuMP
using Ipopt

# Objective function
function NLPModels.obj(nlp :: MyModel, x :: AbstractVector)
    return sum((1 .- x).^2)
  end
  
  # Gradient of the objective function
  function NLPModels.grad!(nlp :: MyModel, x :: AbstractVector, g :: AbstractVector)
    g .= -2 .* (1 .- x)
    return g
  end

tp = mtprw(10, 2, pctzero=.2)
# tp = mtprw(100, 4, pctzero=.3)

model = Model()
@variable(model, x[1:10])
f(x) = sum((1 .- x).^2)
@NLobjective(model, Min, f(x))

n = 5

# Create a JuMP model, using Ipopt as the solver
model = Model(Ipopt.Optimizer)

# Set Ipopt options
hsllib = "/usr/local/lib/lib/x86_64-linux-gnu/libhsl.so"
set_optimizer_attribute(model, "max_iter", 1000)
set_optimizer_attribute(model, "tol", 1e-6)
set_optimizer_attribute(model, "print_level", 5)
set_optimizer_attribute(model, "linear_solver", "ma57")
set_optimizer_attribute(model, "hsllib", hsllib)

# Define the variables with bounds
@variable(model, 0 <= x[1:n] <= 2)

# Define the objective function
@NLobjective(model, Min, sum((1 - x[i])^2 for i in 1:n))

# Solve the problem
optimize!(model)

# Get the solution
x_opt = value.(x)


using NLPModels, NLPModelsIpopt, SparseArrays

# Define the model
mutable struct MyModel <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

# Objective function
function NLPModels.obj(nlp :: MyModel, x :: AbstractVector)
  return sum((1 .- x).^2)
end

# Gradient of the objective function
function NLPModels.grad!(nlp :: MyModel, x :: AbstractVector, g :: AbstractVector)
  g .= -2 .* (1 .- x)
  return g
end

# Hessian structure
function NLPModels.hess_structure!(nlp :: MyModel, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
  rows .= 1:nlp.meta.nvar
  cols .= 1:nlp.meta.nvar
  return rows, cols
end

# Hessian values
function NLPModels.hess_coord!(nlp :: MyModel, x :: AbstractVector, vals :: AbstractVector)
  vals .= 2.0
  return vals
end

# Number of variables
n = 5

# Starting point
x0 = zeros(n)

# Lower and upper bounds
lvar = zeros(n)
uvar = fill(2.0, n)

# Create an instance of MyModel
nlp = MyModel(NLPModelMeta(n, x0=x0, lvar=lvar, uvar=uvar, nnzh=n), Counters())

# Solve the problem using Ipopt
solver = IpoptSolver(print_level=0)
result = solve(solver, nlp, x0)

# Print the solution
println("Solution x = ", result.solution)

using NLPModels, NLPModelsIpopt, SparseArrays

# Define the model
mutable struct MyModel <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

nlp = MyModel(NLPModelMeta(5, x0=zeros(5), lvar=zeros(5), uvar=fill(2.0,5), nnzh=5), Counters())

using NLPModels, NLPModelsIpopt, SparseArrays

# Define the model
mutable struct MyModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
end


mutable struct MyModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    nvar::Int
end

  function MyModel(nvar::Int)
    meta = NLPModelMeta(
      nvar,
      x0=zeros(nvar),
      lvar=zeros(nvar),
      uvar=fill(2.0,nvar),
      nnzh=nvar
    )
    new(meta, Counters(), nvar)
  end

nlp = MyModel(5)

# Objective function
function NLPModels.obj(nlp :: MyModel, x :: AbstractVector)
  return sum((1 .- x).^2)
end

# Gradient of the objective function
function NLPModels.grad!(nlp :: MyModel, x :: AbstractVector, g :: AbstractVector)
  g .= -2 .* (1 .- x)
  return g
end

# Hessian structure
function NLPModels.hess_structure!(nlp :: MyModel, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
  rows .= 1:nlp.nvar
  cols .= 1:nlp.nvar
  return rows, cols
end

# Hessian values
function NLPModels.hess_coord!(nlp :: MyModel, x :: AbstractVector, vals :: AbstractVector)
  vals .= 2.0
  return vals
end

# Solve the problem using Ipopt
solver = IpoptSolver(print_level=0)
result = solve(solver, nlp, nlp.meta.x0)

# Print the solution
println("Solution x = ", result.solution)
