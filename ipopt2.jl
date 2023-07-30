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

# Hessian structure
function NLPModels.hess_structure!(nlp :: MyModel, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
  rows .= 1:nlp.meta.nvar
  cols .= 1:nlp.meta.nvar
  return rows, cols
end

# Hessian values
function NLPModels.hess_coord!(nlp::MyModel, x::AbstractVector, vals::AbstractVector; obj_weight::Real=1.0)
    # As the hessian is constant and doesn't depend on x or y, we ignore these parameters
    vals .= 2.0 * obj_weight
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
res = ipopt(nlp, print_level=5, hessian_constant="yes")

# djb the above works!!

# now add constraints -- the following looks promising but doesn't work yet

using LinearAlgebra
using NLPModels, NLPModelsIpopt, SparseArrays

# Define the model
mutable struct mod2{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    A::Matrix{T}
    b::Vector{T}
end

# Objective function
function NLPModels.obj(nlp :: mod2, x :: AbstractVector)
    return sum((1 .- x).^2)
  end
  
  # Gradient of the objective function
  function NLPModels.grad!(nlp :: mod2, x :: AbstractVector, g :: AbstractVector)
    g .= -2 .* (1 .- x)
    return g
  end


n = 5
A = Matrix{Float64}(I, n, n)
b = ones(n)
nlp = mod2(NLPModelMeta(n, x0=zeros(n), lvar=zeros(n), uvar=fill(2.0,n), nnzh=n, ncon=n, lcon=b, ucon=b), Counters(), A, b) # , nln_nnzj=0

# Hessian structure
function NLPModels.hess_structure!(nlp :: mod2, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
    rows .= 1:nlp.meta.nvar
    cols .= 1:nlp.meta.nvar
    return rows, cols
  end
  
  # Hessian values
  function NLPModels.hess_coord!(nlp::mod2, x::AbstractVector, vals::AbstractVector; obj_weight::Real=1.0)
      # As the hessian is constant and doesn't depend on x or y, we ignore these parameters
      vals .= 2.0 * obj_weight
      return vals
  end


# Constraints
function NLPModels.cons!(nlp::mod2, x::AbstractVector, c::AbstractVector)
    c .= nlp.A * x - nlp.b
    return c
end

# Jacobian structure
# Jacobian of the constraints
function NLPModels.jac_structure!(nlp::mod2, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    rows .= repeat(1:nlp.meta.ncon, nlp.meta.nvar)
    cols .= vcat([fill(i, nlp.meta.ncon) for i = 1:nlp.meta.nvar]...)
    return rows, cols
end

function NLPModels.jac_coord!(nlp::mod2, x::AbstractVector, vals::AbstractVector)
    vals .= nlp.A[:]
    return vals
end


# function NLPModels.jac_nln_structure!(nlp::MyModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
#     @views rows[1:0], cols[1:0] = Int64[], Int64[]
#     return rows, cols
# end

# function NLPModels.jac_nln_coord!(nlp::MyModel, x::AbstractVector, vals::AbstractVector)
#     @views vals[1:0] = Float64[]
#     return vals
# end

# function NLPModels.jac_nln_structure!(nlp::MyModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
#     return rows, cols  # No nonlinear constraints, return empty structure
# end



# Solve the problem using Ipopt
res = ipopt(nlp, print_level=5, hessian_constant="yes")
