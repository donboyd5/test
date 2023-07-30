# Run a series of CUTEst models to test the solver.
#
# Generates the report solvername_cutest.txt that can be used to create a simple
# performance profile comparing solvers. However it should be taken into
# account that the stopping criterion need to be somewhat similar for a
# meaninful comparison.
#
# Obs: It is set to run Algencan now, but you should easily adapt to other
# solvers.

using Printf
using NLPModels
using CUTEst
using NLPModelsAlgencan
using ADNLPModels  # djb

# djb startup
# name = "HS6"
# stats
# fieldnames(typeof(stats))
# fieldnames(typeof(stats.iter))
# finalize(nlp)


function cutest_bench(name, solver)
    nlp = CUTEstModel(name)
    bench_data = @timed stats = solver(nlp)
    etime = bench_data[2]
    flag = stats.status
    objval = obj(nlp, stats.solution)
    # c = stats.counters.counters # djb
    # n_fc, n_ggrad, n_hl, n_hlp = c.neval_obj, c.neval_jac, c.neval_hess, c.neval_hprod # djb
    finalize(nlp)
    # return flag, etime, n_fc, n_ggrad, n_hl, n_hlp, objval
    return flag, etime, objval
end


function run_tests()
    # Algencan tolerances
    solver(model) = algencan(model, epsfeas=1.0e-5, epsopt=1.0e-5, specfnm="algencan.dat")
    solver_name = "algencan_hsl_accel"

    # First run to compile
    set_mastsif()
    cutest_bench("HS6", solver)

    # Grab a list of CUTEst tests
    test_problems = readlines(open("cutest_selection.txt"))
    n_tests = length(test_problems)

    # Run tests
    report = open(string(solver_name, "_cutest.txt"), "w")
    for i = 1:n_tests
        name = test_problems[i]
        println("\nSolving Problem $name - $i of $n_tests.\n")
        s, t, fc, ggrad, hl, hlp, v = cutest_bench(name, solver)

        println("\n*************************************************************")
        println("Problem name = ", name)
        println("Performance = ", t, fc, ggrad, hl, hlp)
        println("Status = ", s)
        println("Obj value = ", v)
        println("*************************************************************\n")
        line = @sprintf("%-14s%-14s%12.4e\t%10.4d\t%10.4d\t%10.4d\t%10.4d\t%12.4e\n",
            name, s, t, fc, ggrad, hl, hlp, v)
        write(report, line)
        flush(report)
    end
    close(report)

    println("Solved ", n_tests, " problems.")
end

run_tests()


function hs52()

    x0 = [2.0; 2.0; 2.0; 2.0; 2.0]
    f(x) = (4 * x[1] - x[2])^2 + (x[2] + x[3] - 2)^2 + (x[4] - 1)^2 + (x[5] - 1)^2
    c(x) = [x[1] + 3 * x[2]; x[3] + x[4] - 2 * x[5]; x[2] - x[5]]
    lcon = [0.0; 0.0; 0.0]
    ucon = [0.0; 0.0; 0.0]
  
    return ADNLPModel(f, x0, c, lcon, ucon, name="hs52_autodiff")
  
  end

hs52()


#Problem 12 in the Hock-Schittkowski suite
function hs12()

    x0 = [0.0, 0.0]
    f(x) = (x[1]^2) / 2 + x[2]^2 - x[1] * x[2] - 7 * x[1] - 7 * x[2]
    c(x) = [-4 * x[1]^2 - x[2]^2]
    lcon = [-25.0]
    ucon = [Inf]
  
    return ADNLPModel(f, x0, c, lcon, ucon, name="hs12_autodiff")
  
  end

m = hs12()
solver(m) = algencan(m, epsfeas=1.0e-5, epsopt=1.0e-5)
solver_name = "algencan_hsl_accel"
# solver(nlp)
solver(m)

    # First run to compile
set_mastsif()
report = open(string(solver_name, "_hs12.txt"), "w")


using NLPModels

# Rosenbrock
nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2 , [-1.2; 1.0])
# ADNLPModel(Minimization problem Generic nvar = 2, ncon = 0 (0 linear), Counters(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), Main.ex-adnlp.var"#1#2"(), NLPModels.var"#133#136"())
fx = obj(nlp, nlp.meta.x0)
println("fx = $fx")

gx = grad(nlp, nlp.meta.x0)
Hx = hess(nlp, nlp.meta.x0)
println("gx = $gx")
println("Hx = $Hx")

using LinearAlgebra

function steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)
  x = nlp.meta.x0
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  slope = dot(∇fx, ∇fx)
  ∇f_norm = sqrt(slope)
  iter = 0
  while ∇f_norm > eps && iter < itmax
    t = 1.0
    x_trial = x - t * ∇fx
    f_trial = obj(nlp, x_trial)
    while f_trial > fx - eta * t * slope
      t *= sigma
      x_trial = x - t * ∇fx
      f_trial = obj(nlp, x_trial)
    end
    x = x_trial
    fx = f_trial
    ∇fx = grad(nlp, x)
    slope = dot(∇fx, ∇fx)
    ∇f_norm = sqrt(slope)
    iter += 1
  end
  optimal = ∇f_norm <= eps
  return x, fx, ∇f_norm, optimal, iter
end

x, fx, ngx, optimal, iter = steepest(nlp)
println("x = $x")
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")

nlp = ADNLPModel(x -> dot(x, x), zeros(2))
for i = 1:100
    obj(nlp, rand(2))
end
neval_obj(nlp)
neval_grad(nlp)

sum_counters(nlp)

bound_constrained(nlp)
equality_constrained(nlp)
nlp.meta


x = zeros(2)
x = [1.3, 2.7]

# Newton
g(x) = grad(nlp, x)
H(x) = Symmetric(hess(nlp, x), :L)
x = nlp.meta.x0
d = -H(x)\g(x)

for i = 1:10
    global x
    x = x - H(x)\g(x)
    println("x = $x")
end

using SparseArrays

using NLPModels, NLPModelsJuMP, JuMP, SparseArrays

# Define the model
model = Model()

# Define the variables with lower and upper bounds
@variable(model, 1 <= x[1:3] <= 5)

# Define the objective function
@NLobjective(model, Min, x[1]^2 + x[2]^2 + x[3]^2)

# Define the constraints
@constraint(model, x[1] + x[2] == 4)
@constraint(model, x[1] + x[3] == 5)
@constraint(model, x[2] + x[3] == 6)

# Convert the JuMP model to an NLPModels type
nlp = MathOptNLPModel(model)

# Inspect the NLP model
println("x0 = ", nlp.meta.x0)
println("lvar = ", nlp.meta.lvar)
println("uvar = ", nlp.meta.uvar)
println("lcon = ", nlp.meta.lcon)
println("ucon = ", nlp.meta.ucon)

using NLPModels

function ADNLPModel(x)
  # Objective function
  @NLPModels.objective(x, 10 * x[1] + 2 * x[2] + 3 * x[3])

  # Constraints
  @NLPModels.constraint(x, 2 * x[1] + x[2] - x[3] <= 1)
  @NLPModels.constraint(x, x[1] + 2 * x[2] - x[3] >= -1)
  @NLPModels.constraint(x, x[1] - x[2] + x[3] == 0)

  # Variable bounds
  @NLPModels.bound(x[1], 0, 1)
  @NLPModels.bound(x[2], -1, 1)
  @NLPModels.bound(x[3], -1, 1)

  return ADNLPModel(x)
end

model = ADNLPModel([1, 2, 3])

# Print the model
println("Model:")
println(model)

# Print the objective function
println("Objective function:")
println(model.objective)

# Print the constraints
println("Constraints:")
for constraint in model.constraints
  println(constraint)
end

# Print the variable bounds
println("Variable bounds:")
for variable in model.variables
  println(variable, ":", model.variable_bounds[variable])
end


using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
set_attribute(model, "max_cpu_time", 60.0)
set_attribute(model, "print_level", 0)


using Ipopt

function ADNLPModel(x::Vector{Float64})

  # Objective function
  @NLPModels.objective(x, 10 * x[1] + 2 * x[2] + 3 * x[3])

  # Constraints
  @NLPModels.constraint(x, 2 * x[1] + x[2] - x[3] <= 1)
  @NLPModels.constraint(x, x[1] + 2 * x[2] - x[3] >= -1)
  @NLPModels.constraint(x, x[1] - x[2] + x[3] == 0)

  # Variable bounds
  @NLPModels.bound(x[1], 0, 1)
  @NLPModels.bound(x[2], -1, 1)
  @NLPModels.bound(x[3], -1, 1)


  # Create a sparse Jacobian object
  sparse_jacobian = SparseJacobian(model)

  # Set the objective function and constraint Jacobians
  sparse_jacobian.f = x -> [10, 2, 3]
  sparse_jacobian.g = x -> [2*x[1]+x[2]-x[3], x[1] + 2*x[2] - x[3], x[1] - x[2] + x[3]]

  return ADNLPModel(x, sparse_jacobian)
end

model = ADNLPModel([1, 2, 3])

# Solve the problem
solver = IpoptSolver(model)
solver.solve()

# Print the solution
println(solver.x) 

using ADNLPModels, NLPModels
using ReverseDiff
using Zygote

function f(x)
    objval = sum(x.^2)
  return objval # , targdiffs, whdiffs, targstop, whstop
end

f([1., 2., 3.])
f([-1.2; 1.0])
f([-1.2, 1.0])



# f2 = (ratio) -> mw.objfn_reweight(ratio, wh, xmat, rwtargets, rweight=0.0)

nlp = ADNLPModel(f, [-1.2; 1.0])
obj(nlp, nlp.meta.x0)
grad(nlp, nlp.meta.x0)
hess(nlp, nlp.meta.x0)

# ADNLPModel(f, x0)
# ADNLPModel(f, x0, lvar, uvar)
# ADNLPModel(f, x0, c, lcon, ucon)
# ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

# https://jso.dev/tutorials/

nlp = ADNLPModel(f, [-1.2, 1.0], lvar=[-5., -6.], uvar=[20., 30.])
obj(nlp, nlp.meta.x0)
grad(nlp, nlp.meta.x0)
hess(nlp, nlp.meta.x0)

x0 = [-1.2; 1.0]
lvar=[-5., -6.]
uvar=[20., 30.]


nlp = ADNLPModel(f, x0)
nlp = ADNLPModel(f, x0, backend=ADNLPModels.ForwardDiffAD())
nlp = ADNLPModel(f, x0; backend = ADNLPModels.ReverseDiffAD)
nlp = ADNLPModel(f, x0; backend = ADNLPModels.ZygoteAD)


using JSOSolvers, ADNLPModels
f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])
a = trunk(nlp, atol = 1e-6, rtol = 1e-6)
fieldnames(typeof(a))
a.status
a.solution
a.objective
a.iter
a.elapsed_time
nlp.counters

# https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl
# https://jso.dev/tutorials/introduction-to-nlpmodelsipopt/

using ADNLPModels, NLPModels, NLPModelsIpopt

nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
stats = ipopt(nlp)
print(stats)

n = 10
x0 = ones(n)
x0 = randn(10)
x0[1:2:end] .= -1.2
lcon = ucon = zeros(n-2)
nlp = ADNLPModel(x -> sum((x[i] - 1)^2 + 100 * (x[i+1] - x[i]^2)^2 for i = 1:n-1), x0,
                 x -> [3 * x[k+1]^3 + 2 * x[k+2] - 5 + sin(x[k+1] - x[k+2]) * sin(x[k+1] + x[k+2]) +
                       4 * x[k+1] - x[k] * exp(x[k] - x[k+1]) - 3 for k = 1:n-2],
                 lcon, ucon)
stats = ipopt(nlp, print_level=5)
print(stats)
stats.solver_specific[:internal_msg]
