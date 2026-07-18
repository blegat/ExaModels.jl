# Convert PowerModels OPF instances, in various formulations, to ExaModels through
# the MOI extension, and report the conversion time together with the number of
# constraint kernels detected by the structure re-discovery (the `Bin` grouping in
# ext/ExaModelsMOI.jl). A kernel count that grows with the instance size means the
# grouping failed and compilation time will blow up.
#
# Usage:
#     julia --project=benchmark/powermodels benchmark/powermodels/kernels.jl [case files...]
#
# Without arguments, the matpower cases shipped with PowerModels are used.
using JuMP
import ExaModels
import MathOptInterface as MOI
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
    end
    flush(io)
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
