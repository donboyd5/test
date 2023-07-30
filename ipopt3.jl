# https://jso.dev/NLPModels.jl/stable/guidelines/


using LinearAlgebra
using NLPModels, NLPModelsIpopt, SparseArrays

hsllib = "/usr/local/lib/lib/x86_64-linux-gnu/libhsl.so"

# Define the model
mutable struct mod2{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    A::Matrix{T}
    b::Vector{T}
end

# Objective function
function NLPModels.obj(nlp::mod2, x::AbstractVector)
    return sum((1 .- x).^2)
  end
  
  # Gradient of the objective function
  function NLPModels.grad!(nlp::mod2, x::AbstractVector, g::AbstractVector)
    g .= -2 .* (1 .- x)
    return g
  end


# Hessian structure
function NLPModels.hess_structure!(nlp::mod2, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
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
# function NLPModels.jac_structure!(nlp::mod2, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
#     rows .= repeat(1:nlp.meta.ncon, nlp.meta.nvar)
#     cols .= vcat([fill(i, nlp.meta.ncon) for i = 1:nlp.meta.nvar]...)
#     return rows, cols
# end

function NLPModels.jac_structure!(nlp::mod2, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    rows .= 1:nlp.meta.nvar
    cols .= 1:nlp.meta.nvar
    return rows, cols
  end

# function NLPModels.jac_coord!(nlp::mod2, x::AbstractVector, vals::AbstractVector)
#     vals .= nlp.A[:]
#     return vals
# end

function NLPModels.jac_coord!(nlp::mod2, x::AbstractVector, vals::AbstractVector)
    vals .= nlp.A * x
    return vals
end


n = 5
A = Matrix{Float64}(I, n, n)
b = ones(n)
# x0=zeros(n) # not needed
# A * x0

# first, specify without constraints
nlp = mod2(NLPModelMeta(n, x0=zeros(n), lvar=zeros(n), uvar=fill(2.0,n), nnzh=n), Counters(), A, b)

# now try to impose constraints
nlp2 = mod2(NLPModelMeta(n, x0=zeros(n), lvar=zeros(n), uvar=fill(2.0,n), nnzh=n, ncon=n, lcon=b, ucon=b, nln_nnzj=0), Counters(), A, b) # , nln_nnzj=0

# Solve the problem using Ipopt
res = ipopt(nlp, print_level=5, hessian_constant="yes")
res = ipopt(nlp, print_level=5, hessian_constant="yes", hsllib=hsllib, linear_solver="ma57")



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

mutable struct MyModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    A::Matrix{T}
    b::Vector{T}
end

function MyModel(nvar::Int, 
    A::Matrix{T}, b::Vector{T};      
    x0::Vector{T}=zeros(nvar), lvar::Vector{T}=fill(-Inf, nvar), uvar::Vector{T}=fill(Inf, nvar),
    ncon::Int=0, lcon::Vector{T}=fill(-Inf, ncon), ucon::Vector{T}=fill(Inf, ncon), nln_nnzj::Int=nvar,
    nnzj::Int=0, nnzh::Int=nvar) where T

    # Create the NLPModelMeta instance
    meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, lcon=lcon, ucon=ucon, 
            nnzj=nnzj, nnzh=nnzh, nln_nnzj=nln_nnzj)

    # Create the Counters instance
    counters = Counters()

    # Create and return the modMyModel instance
    return MyModel(meta, counters, A, b)
end

n = 5
A = Matrix{Float64}(I, n, n)
b = ones(n)
mod1 = MyModel(5, A, b)
mod1.meta.nvar
mod1.meta.x0
mod1.meta.lvar


# Objective function
function NLPModels.obj(nlp::MyModel, x::AbstractVector)
    return sum((1 .- x).^2)
end
  
  # Gradient of the objective function
function NLPModels.grad!(nlp::MyModel, x::AbstractVector, g::AbstractVector)
    g .= -2 .* (1 .- x)
    return g
end


# Hessian structure
function NLPModels.hess_structure!(nlp::MyModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
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

res = NLPModelsIpopt.ipopt(mod1, print_level=5, hessian_constant="yes")

res = NLPModelsIpopt.ipopt(mod1, print_level=5, hessian_constant="yes", hsllib=hsllib, linear_solver="ma57")


# add constraints --------------------------------------------------------------------------------------------------
using LinearAlgebra
using NLPModels, NLPModelsIpopt, SparseArrays

hsllib = "/usr/local/lib/lib/x86_64-linux-gnu/libhsl.so"



mutable struct mmcon{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    A::Matrix{T}
    b::Vector{T}
end

function mmcon(nvar::Int, 
    A::Matrix{T}, b::Vector{T};      
    x0::Vector{T}=zeros(nvar), lvar::Vector{T}=fill(-Inf, nvar), uvar::Vector{T}=fill(Inf, nvar),
    ncon::Int=0, lcon::Vector{T}=fill(-Inf, ncon), ucon::Vector{T}=fill(Inf, ncon), lin::UnitRange{Int64}=1:ncon,
    nnzj::Int=0, nnzh::Int=nvar) where T

    # Create the NLPModelMeta instance
    meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, lcon=lcon, ucon=ucon, 
            nnzj=nnzj, nnzh=nnzh, lin=lin)

    # Create the Counters instance
    counters = Counters()

    # Create and return the modMyModel instance
    return mmcon(meta, counters, A, b)
end

n = 5
A = Matrix{Float64}(I, n, n)
b = ones(n)
A * b
mod2 = mmcon(5, A, b, ncon=5, lcon=b, ucon=b, nnzj=5)

# obj(nlp, x)
# grad!(nlp, x, g)
# hess_structure!(nlp, hrows, hcols)
# hess_coord!(nlp, x, hvals; obj_weight=1) # unconstrained
# hess_coord!(nlp, x, y, hvals; obj_weight=1) # constrained -- needs y

# cons_lin!(nlp, x, c)  # c = cons_lin(nlp, x)
# jac_lin_structure!(nlp, jrows, jcols)
# jac_lin_coord!(nlp, x, jvals)
# hess_coord!(nlp, x, y, hvals; obj_weight=1)



# Objective function
function NLPModels.obj(nlp::mmcon, x::AbstractVector)
    return sum((1 .- x).^2)
end
  
  # Gradient of the objective function
function NLPModels.grad!(nlp::mmcon, x::AbstractVector, g::AbstractVector)
    g .= -2 .* (1 .- x)
    return g
end

# Hessian structure
function NLPModels.hess_structure!(nlp::mmcon, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    rows .= 1:nlp.meta.nvar
    cols .= 1:nlp.meta.nvar
    return rows, cols
  end
  
  # Hessian values
  function NLPModels.hess_coord!(nlp::mmcon, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight::Real=1.0)
      # As the hessian is constant and doesn't depend on x or y, we ignore these parameters
      vals .= 2.0 * obj_weight
      return vals
  end


# constraints  
function NLPModels.cons_lin!(nlp::mmcon, x::AbstractVector, c::AbstractVector)
    c .= nlp.A * x - nlp.b
    return c
end  

# Jacobian structure
function NLPModels.jac_structure!(nlp::mmcon, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    # If the constraints are linear, the Jacobian structure is the same as the A matrix non-zero structure
    r, c = findnz(sparse(nlp.A))
    rows .= r
    cols .= c
    return rows, cols
end


# Jacobian values
function NLPModels.jac_coord!(nlp::mmcon, x::AbstractVector, vals::AbstractVector)
    # If the constraints are linear, the Jacobian values are the same as the A matrix non-zero values
    vals .= nonzeros(sparse(nlp.A))
    return vals
end



res = ipopt(mod2, print_level=5, hessian_constant="yes")

res = ipopt(mod2, print_level=5, hessian_constant="yes", hsllib=hsllib, linear_solver="ma57")




# test the model --------------------------------------------------------------------------------------
NLPModels.obj(mod1, mod1.meta.x0)

g = [0., 0., 0., 0., 0.]
NLPModels.grad!(mod1, mod1.meta.x0, g)

println("Size of A: ", size(mod1.A))
println("Length of b: ", length(mod1.b))
println("Number of variables: ", mod1.meta.nvar)
println("Number of constraints: ", mod1.meta.ncon)
println("Size of x0: ", length(mod1.meta.x0))
println("Size of lvar: ", length(mod1.meta.lvar))
println("Size of uvar: ", length(mod1.meta.uvar))

x = rand(mod1.meta.nvar)
g = zeros(mod1.meta.nvar)
NLPModels.grad!(mod1, x, g)
println("Size of gradient: ", length(g))

rows = zeros(Int, mod1.meta.nnzh)
cols = zeros(Int, mod1.meta.nnzh)
vals = zeros(mod1.meta.nnzh)
NLPModels.hess_structure!(mod1, rows, cols)
NLPModels.hess_coord!(mod1, x, vals)
println("Size of Hessian structure: ", length(rows), ", ", length(cols))
println("Size of Hessian values: ", length(vals))




rows = 1:mod1.meta.nvar
cols = 1:mod1.meta.nvar
NLPModels.hess_structure!(mod1, rows, cols)

# function MyModel(nvar::Int, ncon::Int, x0::Vector{T}, lvar::Vector{T}, uvar::Vector{T}, 
#     lcon::Vector{T}, ucon::Vector{T}, A::Matrix{T}, b::Vector{T}, nnzj::Int, 
#     nnzh::Int) where T

#     # Create the NLPModelMeta instance
#     meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, lcon=lcon, ucon=ucon, 
#             nnzj=nnzj, nnzh=nnzh)

#     # Create the Counters instance
#     counters = Counters()

#     # Create and return the modMyModel instance
#     return modMyModel(meta, counters, A, b)
# end

mat = [8 0 7; 0 9 0]
rows, cols, vals = findnz(sparse(mat))

