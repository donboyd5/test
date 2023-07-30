using DZOptimization
using ForwardDiff

include("mtprw.jl")


function box_constraint!(point::Array{Float64,N}) where {N}
    @simd ivdep for i = 1 : length(point)
        @inbounds point[i] = clamp(point[i], lb, ub)
    end
    return true
end  


function objfn_reweight(ratio, wh, xmat, rwtargets, rweight)
  
    # part 1 get measure of difference from targets
    rwtargets_calc = xmat' * (ratio .* wh)
    targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
    ss_targpdiffs = sum(targpdiffs.^2.)
    avg_tdiff = ss_targpdiffs / length(targpdiffs)
  
    # part 2 - measure of change in ratio
    ratiodiffs = ratio .- 1.0
    ss_ratiodiffs = sum(ratiodiffs.^6.)
    avg_rdiff = ss_ratiodiffs / length(ratiodiffs)
  
    # combine the two measures and (maybe later) take a root
    # objval = (ss_targdiffs / length(targdiffs))*(1. - whweight) +
    #         (ss_whdiffs / length(whdiffs))*whweight
    # objval = objval^(1. / pow)  
    # objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight
    objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight
  
    # list extra variables on the return so that they are available to the callback function
    return objval # , targdiffs, whdiffs, targstop, whstop
  end



f = (ratio) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight)

function g!(grad::Vector{T}, x::Vector{T}) where T
    # Compute the gradient of f(x) and store it in grad (inplace)
    ForwardDiff.gradient!(grad, f, x)
end



h = 100_000
k = 20
pzero = 0.2

tp = mtprw(h, k, pctzero=pzero)
wh = tp.wh
xmat = tp.xmat
rwtargets = tp.rwtargets
rweight = 0.1

x = ones(tp.h)
lb = 0.1
ub = 2.0

f(x)
pdiffs0 = tp.rwtargets_calc ./ tp.rwtargets .- 1
quantile(pdiffs0)


opt = LBFGSOptimizer(f,
    g!,
    box_constraint!,
    x,
    1.0)   

obj = 1e9    
change = 1e9
while (!opt.has_converged[]) && (opt.current_objective_value[] > 1e-4) && (opt.iteration_count[] < 100) && (change > 1e-5)
    println(opt.iteration_count[], '\t', opt.current_objective_value[])    
    change = obj - opt.current_objective_value[]
    obj = opt.current_objective_value[]
    step!(opt)
end  

fieldnames(typeof(opt))
opt.iteration_count
opt.has_converged
opt.delta_point
xsol = opt.current_point

quantile(xsol)

rwtargets_calc = xmat' * (xsol .* wh)
targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets 
quantile(targpdiffs)

pdiffs0 = tp.rwtargets_calc ./ tp.rwtargets .- 1
quantile(pdiffs0)


rosenbrock_objective(x::Vector) =
    (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

function rosenbrock_gradient!(g::Vector, x::Vector)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
end

opt = BFGSOptimizer(rosenbrock_objective,
                    rosenbrock_gradient!,
                    rand(2), # starting point
                    1.0)     # initial step size

opt = LBFGSOptimizer(rosenbrock_objective,
    rosenbrock_gradient!,
    [0., 0.],
    # rand(2), # starting point
    1.0)   

    
function box_constraint!(point::Array{Float64,N}) where {N}
    @simd ivdep for i = 1 : length(point)
        @inbounds point[i] = clamp(point[i], 0.0, 0.6)
    end
    return true
end    

function box_constraint!(point::Array{Float64,N}) where {N}
    @simd ivdep for i = 1 : length(point)
        @inbounds point[i] = clamp(point[i], lb, ub)
    end
    return true
end  

lb = 0.0
ub = 0.4

opt = LBFGSOptimizer(rosenbrock_objective,
    rosenbrock_gradient!,
    box_constraint!,
    [0.5, 0.5],
    # rand(2), # starting point
    1.0)   

while !opt.has_converged[]
    # println(opt.iteration_count[], '\t', opt.constraint_function![], '\t', opt.current_objective_value[], '\t', opt.current_point)
    println(opt.iteration_count[], '\t', opt.current_objective_value[], '\t', opt.current_point)
    step!(opt)
end    


opt = GradientDescentOptimizer(rosenbrock_objective,
    rosenbrock_gradient!,
    [0., 0.],
    # rand(2), # starting point
    1.0)       
        

fieldnames(typeof(opt))
opt.objective_function
opt.iteration_count
opt.current_objective_value

function box_constraint!(point::Array{Float64,N}) where {N}
    @simd ivdep for i = 1 : length(point)
        @inbounds point[i] = clamp(point[i], 0.0, 0.6)
    end
    return true
end    


while !opt.has_converged[]
    # println(opt.iteration_count[], '\t', opt.constraint_function![], '\t', opt.current_objective_value[], '\t', opt.current_point)
    println(opt.iteration_count[], '\t', opt.current_objective_value[], '\t', opt.current_point)
    step!(opt)
end


function LBFGSOptimizer(objective_function::F,
    gradient_function!::G,
    constraint_function!::C,
    initial_point::AbstractArray{T,N},
    initial_step_size::T,
    history_length::Int=10) where {F,G,C,T,N}
@assert history_length > 0
iteration_count = fill(0)
has_converged = fill(false)
current_point = copy(initial_point)
constraint_success = constraint_function!(current_point)
@assert constraint_success
initial_objective_value = objective_function(current_point)
@assert !isnan(initial_objective_value)
current_objective_value = fill(initial_objective_value)
current_gradient = similar(initial_point)
gradient_function!(current_gradient, current_point)
delta_point = zero(initial_point)
delta_gradient = zero(initial_point)
last_step_size = fill(initial_step_size)
next_step_direction = scalar_mul!(copy(current_gradient),
initial_step_size / norm(current_gradient))
_alpha_history = zeros(T, history_length)
_rho_history = zeros(T, history_length)
_delta_point_history = zeros(T, length(initial_point), history_length)
_delta_gradient_history = zeros(T, length(initial_point), history_length)
_line_search_functor = LineSearchFunctor{F,C,T,N}(
objective_function, constraint_function!,
current_point, similar(initial_point), next_step_direction)
return LBFGSOptimizer{F,G,C,T,N}(
objective_function,
gradient_function!,
constraint_function!,
iteration_count,
has_converged,
current_point,
current_objective_value,
current_gradient,
delta_point,
delta_gradient,
last_step_size,
next_step_direction,
_alpha_history,
_rho_history,
_delta_point_history,
_delta_gradient_history,
_line_search_functor)
end


# mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[1], rweight=rwt, scaling=scaleit, maxit=iters)



function objfn_reweight(
    ratio, wh, xmat, rwtargets;
    rweight=0.5,
    pow=2.0,
    targstop=true, whstop=true,
    display_progress=true)
  
    # part 1 get measure of difference from targets
    rwtargets_calc = xmat' * (ratio .* wh)
    targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
    ss_targpdiffs = sum(targpdiffs.^2.)
    avg_tdiff = ss_targpdiffs / length(targpdiffs)
  
    # part 2 - measure of change in ratio
    ratiodiffs = ratio .- 1.0
    ss_ratiodiffs = sum(ratiodiffs.^6.)
    avg_rdiff = ss_ratiodiffs / length(ratiodiffs)
  
    # combine the two measures and (maybe later) take a root
    # objval = (ss_targdiffs / length(targdiffs))*(1. - whweight) +
    #         (ss_whdiffs / length(whdiffs))*whweight
    # objval = objval^(1. / pow)  
    # objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight
    objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight
  
    # list extra variables on the return so that they are available to the callback function
    return objval # , targdiffs, whdiffs, targstop, whstop
  end

  

