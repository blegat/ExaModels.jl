module ExaModelsGenOpt

import ExaModels
import GenOpt
import GenOpt: FunctionGenerator, SumGenerator, ContiguousArrayOfVariables, IteratorIndex, Iterator
import MathOptInterface as MOI

# Mark GenOpt function types as extension types
ExaModels.is_extension_type(::Type{<:FunctionGenerator}) = true
ExaModels.is_extension_type(::Type{<:SumGenerator}) = true

function _map_indices(index_map, f::MOI.ScalarNonlinearFunction)
    args = Any[_map_indices(index_map, arg) for arg in f.args]
    return MOI.ScalarNonlinearFunction(f.head, args)
end

function _map_indices(index_map, array::ContiguousArrayOfVariables)
    first_src = MOI.VariableIndex(array.offset + 1)
    first_dest = _map_variable(index_map, first_src)
    return ContiguousArrayOfVariables(first_dest.value - 1, array.size)
end

_map_indices(::Any, arg) = arg

_map_variable(index_map::MOI.Utilities.IndexMap, variable) = index_map[variable]
_map_variable(index_map::Function, variable) = index_map(variable)

function MOI.Utilities.map_indices(
    index_map::MOI.Utilities.IndexMap,
    f::FunctionGenerator{F},
) where {F}
    return FunctionGenerator{F}(_map_indices(index_map, f.func), f.iterators)
end


function MOI.Utilities.map_indices(
    index_map::Function,
    f::FunctionGenerator{F},
) where {F}
    return FunctionGenerator{F}(_map_indices(index_map, f.func), f.iterators)
end

function MOI.Utilities.map_indices(
    index_map::Function,
    f::SumGenerator{F},
) where {F}
    return SumGenerator{F}(_map_indices(index_map, f.func), f.iterators)
end

function MOI.Utilities.map_indices(
    index_map::MOI.Utilities.IndexMap,
    f::SumGenerator{F},
) where {F}
    return SumGenerator{F}(_map_indices(index_map, f.func), f.iterators)
end

# Handle SumGenerator in objective expressions
function ExaModels.exafy_extension_obj_arg(m::SumGenerator)
    return _exagen(m.func, m.iterators)
end

function ExaModels.add_extra_constraint!(model, f::FunctionGenerator, s)
    exa_moi = Base.get_extension(ExaModels, :ExaModelsMOI)
    row = length(model.lcon) + 1
    expr, pars = _exagen(f.func, f.iterators)
    indexed = ExaModels.DataIndexed(ExaModels.DataSource(), length(first(pars)) + 1)
    data = [(p..., row + i - 1) for (i, p) in enumerate(pars)]
    push!(model.cons, exa_moi.Bin(indexed => expr, data))
    append!(model.lcon, _lower_bounds(s, eltype(model.lcon)))
    append!(model.ucon, _upper_bounds(s, eltype(model.ucon)))
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(row)
end

# Convert GenOpt expression trees to ExaModels format

exagen(α::Number, _) = α

function exagen(f::MOI.ScalarNonlinearFunction, offsets)
    if f.head == :getindex
        v = f.args[1]
        if v isa ContiguousArrayOfVariables
            idx = exagen(f.args[2], offsets)
            if !iszero(v.offset)
                idx = v.offset + idx
            end
            cp = cumprod(v.size)
            for i in 3:length(f.args)
                idx += cp[i - 2] * (exagen(f.args[i], offsets) - 1)
            end
            return ExaModels.Var(idx)
        elseif v isa IteratorIndex
            @assert length(f.args) == 2
            @assert f.args[2] isa Integer
            if isnothing(offsets)
                @assert isone(f.args[2])
                return ExaModels.DataSource()
            else
                return ExaModels.DataIndexed(ExaModels.DataSource(), offsets[v.value] + f.args[2])
            end
        else
            error("Unexpected the first operand of `getindex` to be of type `$(typeof(v))`")
        end
    else
        op = getfield(MOI.Nonlinear, f.head)
        return op((exagen(e, offsets) for e in f.args)...)
    end
end

function _exagen(func::MOI.ScalarNonlinearFunction, iterators)
    lengths = map(it -> length(first(it.values)), iterators)
    cs = [0; cumsum(lengths)[1:(end - 1)]]
    pars = vec(
        map(Base.Iterators.ProductIterator(ntuple(i -> iterators[i].values, length(iterators)))) do I
            reduce((i, j) -> tuple(i..., j...), I)
        end
    )
    expr = exagen(func, cs)
    return expr, pars
end

# Bound helpers for vector sets used by FunctionGenerator constraints
_lower_bounds(s::Union{MOI.Zeros,MOI.Nonnegatives}, T) = fill(zero(T), MOI.dimension(s))
_lower_bounds(s::MOI.Nonpositives, T) = fill(typemin(T), MOI.dimension(s))
_upper_bounds(s::Union{MOI.Zeros,MOI.Nonpositives}, T) = fill(zero(T), MOI.dimension(s))
_upper_bounds(s::MOI.Nonnegatives, T) = fill(typemax(T), MOI.dimension(s))

end # module
