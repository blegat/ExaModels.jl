# Convert PowerModels OPF instances, in various formulations, to ExaModels through
# the MOI extension, and report the conversion time together with the number of
# constraint kernels detected by the structure re-discovery (the `Bin` grouping in
# ext/ExaModelsMOI.jl). A kernel count that grows with the instance size means the
# grouping failed and compilation time will blow up.
#
# Each formulation is then solved twice with Ipopt limited to a single iteration
# (so the timings measure the model copy, the AD setup and one evaluation round,
# not Ipopt's convergence): once with `Ipopt.Optimizer` (JuMP's own AD) and once
# with `ExaModels.Optimizer(NLPModelsIpopt.ipopt)` (this MOI extension).
#
# Usage:
#     julia --project=benchmark/powermodels benchmark/powermodels/kernels.jl [pglib case name]
using JuMP
import ExaModels
import Ipopt
import MathOptInterface as MOI
import NLPModelsIpopt
import PowerModels
import PGLib
import Printf

const ExaMOI = Base.get_extension(ExaModels, :ExaModelsMOI)

const FORMULATIONS = [
    PowerModels.ACPPowerModel,   # polar NLP
    PowerModels.ACRPowerModel,   # rectangular NLP
    PowerModels.ACTPowerModel,   # w-theta NLP
    PowerModels.DCPPowerModel,   # DC LP
    PowerModels.SOCWRPowerModel, # SOC relaxation (quadratic)
    PowerModels.QCRMPowerModel,  # QC relaxation
    PowerModels.LPACCPowerModel, # LP AC cold-start approximation
]

function report(io, data, F)
    pm = PowerModels.instantiate_model(data, F, PowerModels.build_opf)
    moim = JuMP.backend(pm.model).model_cache
    core = try
        # first call: conversion + compilation of the kernel types
        t_cold = @elapsed core, _ = ExaMOI.to_exacore(moim)
        # second call: same kernel types, pure conversion runtime
        t_warm = @elapsed ExaMOI.to_exacore(moim)
        Printf.@printf(
            io,
            "  %-18s nvar=%-6d ncon=%-6d con kernels=%-3d obj kernels=%-3d cold=%6.2fs warm=%6.2fs\n",
            nameof(F), core.nvar, core.ncon, length(core.cons), length(core.obj), t_cold, t_warm,
        )
        core
    catch err
        Printf.@printf(io, "  %-18s FAILED: %s\n", nameof(F), sprint(showerror, err))
        nothing
    end
    if !isnothing(core)
        sizes = sort!([length(c.itr) for c in core.cons]; rev = true)
        println(io, " " ^ 21, "|I| = ", sizes)
        t_jump, o_jump = _timed_solve(data, F, Ipopt.Optimizer)
        t_exa, o_exa =
            _timed_solve(data, F, () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt))
        Printf.@printf(
            io,
            "%ssolve (max_iter = 1): Ipopt.Optimizer =%7.2fs (obj %.6e)  ExaModels+ipopt =%7.2fs (obj %.6e)\n",
            " " ^ 21, t_jump, o_jump, t_exa, o_exa,
        )
    end
    flush(io)
end

# Time a full solve — `optimize!` includes the copy to the solver, the AD setup
# and, with `max_iter = 1`, a single Ipopt iteration.
function _timed_solve(data, F, optimizer)
    pm = PowerModels.instantiate_model(data, F, PowerModels.build_opf)
    model = pm.model
    JuMP.set_optimizer(model, optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", 0)
    t = @elapsed JuMP.optimize!(model)
    return t, JuMP.objective_value(model)
end

function main(name)
    PowerModels.silence()
    data = PGLib.pglib(name)
    PowerModels.standardize_cost_terms!(data, order = 2)
    PowerModels.calc_thermal_limits!(data)
    println(stdout, name, " (", length(data["bus"]), " buses)")
    for F in FORMULATIONS
        report(stdout, data, F)
    end
end
