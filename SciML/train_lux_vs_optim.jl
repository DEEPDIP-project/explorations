# """ 
# We compare here two ways to train Lux models: the old way using `Optimization` and the new way using `Lux`.
# Both of them are available in CoupledNODE.jl
#
# To use this script you have to place it in /simulations/NavierStokes_2D/scripts/ and run it from there.
# """

using CoupledNODE: cnn, train, callback, create_loss_post_lux
using CoupledNODE.NavierStokes: create_right_hand_side_with_closure
using DifferentialEquations: ODEProblem, solve, Tsit5
using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
using JLD2: @save
using Optimization: Optimization
using OptimizationOptimisers: OptimizationOptimisers
using Random: Random

T = Float32
ArrayType = Array
rng = Random.Xoshiro(123)
ig = 1 # index of the LES grid to use.
include("preprocess_posteriori.jl")

# * Creation of the model: NN closure
closure, θ, st = cnn(;
    setup = setups[ig],
    radii = [3, 3],
    channels = [2, 2],
    activations = [tanh, identity],
    use_bias = [false, false],
    rng
)

# * Define the right hand side of the ODE
dudt_nn2 = create_right_hand_side_with_closure(
    setups[ig], INS.psolver_spectral(setups[ig]), closure, st)

# * Define the loss (a-posteriori) - old way
using CoupledNODE: loss_posteriori_optim

# Set some parameters to have a fair comparison
NEPOCHS = 100
LR = 1e-2
NUSE_VAL = 100

# * train a-posteriori: old way single data point
train_data_posteriori = dataloader_posteriori()
optf = Optimization.OptimizationFunction(
    (x, _) -> loss_posteriori_optim(dudt_nn2, x, st, dataloader_posteriori), # x here is the optimization variable (θ params of NN)
    Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)
optim_result, optim_t, optim_mem, _ = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(LR);
    callback = callback,
    maxiters = NEPOCHS,
    progress = true
)
θ_posteriori_optim = optim_result.u

# * package loss
loss_posteriori_lux = create_loss_post_lux(dudt_nn2; sciml_solver = Tsit5())
loss_posteriori_lux(closure, θ, st, train_data_posteriori)

# * training via Lux
lux_result, lux_t, lux_mem, _ = @timed train(
    closure, θ, st, dataloader_posteriori, loss_posteriori_lux;
    nepochs = NEPOCHS, ad_type = Optimization.AutoZygote(),
    alg = OptimizationOptimisers.Adam(LR), cpu = true, callback = callback)

loss, tstate = lux_result
# the trained params are:
θ_posteriori_lux = tstate.parameters

# * validate the results
using CoupledNODE: validate_results
val_optim = validate_results(dudt_nn2, θ_posteriori_optim, dataloader_posteriori, NUSE_VAL)
val_lux = validate_results(dudt_nn2, θ_posteriori_lux, dataloader_posteriori, NUSE_VAL)

# Plot the comparison between the two training methods
using Plots
p1 = bar(["Optim", "Lux"], [optim_t, lux_t], ylabel = "Time (s)",
    title = "Training time", legend = false)
p2 = bar(["Optim", "Lux"], [optim_mem, lux_mem], ylabel = "Memory (MB)",
    yaxis = :log, title = "Memory usage", legend = false)
p3 = bar(["Optim", "Lux"], [val_optim, val_lux], ylabel = "Loss",
    yaxis = :log, title = "Validation loss", legend = false)
plot(p1, p2, p3, layout = (3, 1), size = (600, 700),
    suptitle = "Comparison between loss backends")

# The figure above shows that the Lux backend is faster and has a similar accuracy to the Optim backend. Then why should we use the Optim backend?
# The Optim backend is more flexible and can be used with any optimization algorithm, like the following example:
optf = Optimization.OptimizationFunction(
    (x, _) -> loss_posteriori_optim(closure, x, st, dataloader_posteriori),
    Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ)
optim_result, optim_t, optim_mem, _ = @timed Optimization.solve(
    optprob,
    Optimization.LBFGS();
    callback = callback,
    maxiters = NEPOCHS,
    progress = true
)
# some fix is required here. Look at the previous examples using GrayScott models.