module ExaModelsGenOpt

import ExaModels
import GenOpt
import GenOpt:
    FunctionGenerator,
    SumGenerator,
    FilteredSumGenerator,
    FilterExpression,
    IteratorValues,
    ContiguousArrayOfVariables,
    IteratorIndex,
    Iterator
import MathOptInterface as MOI

# Mark GenOpt function types as extension types
ExaModels.is_extension_type(::Type{<:FunctionGenerator}) = true
ExaModels.is_extension_type(::Type{<:SumGenerator}) = true
ExaModels.is_extension_type(::Type{<:FilteredSumGenerator}) = true

# Handle SumGenerator in objective expressions
function ExaModels.exafy_extension_obj_arg(m::SumGenerator, var_to_idx)
    return _exagen(m.func, m.iterators, var_to_idx)
end

# Handle `SumGenerator`/`FilteredSumGenerator` terms found inside `ScalarNonlinearFunction`
# constraints. `copy_constraints!` builds the base constraint block `cons` (all non-generator
# terms) and hands us the generator terms as `(row, pos, term)` tuples; here we add each one
# as an `add_con!` augmentation of `cons`, exactly like the `@add_con!` augmentation used in
# the ExaModels OPF tutorial. `row` is the (local, 1-based) constraint row within `cons`,
# `pos` the sign.
function ExaModels.copy_extension_con_augmentations!(c, cons, deferred, var_to_idx)
    for (row, pos, term) in deferred
        c = _augment_constraint!(c, cons, row, pos, term, var_to_idx)
    end
    return c
end

function _augment_constraint!(
    c,
    cons,
    row::Int,
    pos::Bool,
    gen::Union{SumGenerator,FilteredSumGenerator},
    var_to_idx,
)
    # Only the simple single-iterator case is supported (as used in the OPF example).
    @assert length(gen.iterators) == 1 "Only single-iterator generators are supported in constraints"
    expr, pars = _exagen(gen.func, gen.iterators, var_to_idx)
    if gen isa FilteredSumGenerator
        pars = _filter_pars(gen.filter, pars)
    end
    isempty(pars) && return c
    # `row` is a constant within this per-row constraint, so the augmentation adds each
    # (filtered) term to that single row: `add_con!(c, cons, row => term for term in pars)`.
    signed = pos ? expr : -expr
    c, _ = ExaModels.add_con!(c, cons, row => signed for _ in pars)
    return c
end

# Select the parameter tuples that satisfy an equality filter `iterator == constant`
# (e.g. `if arc_bus[j] == i`). `IteratorValues.value_index` is the position of the compared
# value inside each flattened parameter tuple produced by `_exagen`.
function _filter_pars(filter::FilterExpression, pars)
    @assert filter.head == :(==) "Only equality (`==`) filters are supported in constraints"
    @assert length(filter.args) == 2 "Only binary equality filters are supported"
    a, b = filter.args
    iv, target = a isa IteratorValues ? (a, b) : (b, a)
    @assert iv isa IteratorValues "Filter must compare an iterator value to a constant"
    @assert iv.index.value == 1 "Only single-iterator filters are supported"
    vidx = iv.value_index
    return [p for p in pars if p[vidx] == target]
end

# Hook to process FunctionGenerator constraints after standard constraints
function ExaModels.copy_extra_constraints!(c, moim, var_to_idx, con_to_idx, T)
    con_types = MOI.get(moim, MOI.ListOfConstraintTypesPresent())
    for (F, S) in con_types
        F <: FunctionGenerator || continue
        cis = MOI.get(moim, MOI.ListOfConstraintIndices{F, S}())
        c = _copy_generator_constraints!(c, moim, cis, var_to_idx, con_to_idx, T, S)
    end
    return c
end

function _copy_generator_constraints!(c, moim, cis, var_to_idx, con_to_idx, T, ::Type{S}) where {S}
    for ci in cis
        func = MOI.get(moim, MOI.ConstraintFunction(), ci)
        set = MOI.get(moim, MOI.ConstraintSet(), ci)
        con_to_idx[ci] = c.ncon
        # Split off any `SumGenerator`/`FilteredSumGenerator` nested at an additive position
        # (e.g. `lazy_sum(... if ...)` inside a `container = ParametrizedArray` power balance).
        # The base block is built with `add_con`; each generator is then applied as an
        # `add_con!` augmentation, exactly like `@add_con(c9, ...)` + `@add_con!(c9, a.bus => p[a.i] ...)`
        # in the ExaModels OPF tutorial.
        gens = Tuple{Bool,Any}[]
        base = _extract_gens(func.func, true, gens)
        expr, pars = _exagen(base, func.iterators, var_to_idx)
        c, cons = ExaModels.add_con(c, (expr for p in pars); lcon = _lower_bounds(set, T), ucon = _upper_bounds(set, T))
        for (pos, gen) in gens
            c = _augment_function_constraint!(c, cons, pos, gen, func.iterators, var_to_idx)
        end
    end
    return c
end

# Replace each additive `SumGenerator`/`FilteredSumGenerator` in `f` by `0.0`, recording
# `(pos, generator)` (with the additive sign) in `gens`, and return the resulting base template.
function _extract_gens(f, pos, gens)
    if f isa SumGenerator || f isa FilteredSumGenerator
        push!(gens, (pos, f))
        return 0.0
    elseif f isa MOI.ScalarNonlinearFunction && f.head == :+
        return MOI.ScalarNonlinearFunction(:+, Any[_extract_gens(a, pos, gens) for a in f.args])
    elseif f isa MOI.ScalarNonlinearFunction && f.head == :-
        args = Any[_extract_gens(f.args[1], pos, gens)]
        for k in 2:length(f.args)
            push!(args, _extract_gens(f.args[k], !pos, gens))
        end
        return MOI.ScalarNonlinearFunction(:-, args)
    else
        return f
    end
end

# Apply a `FilteredSumGenerator` found inside a `FunctionGenerator` (constraint block `cons`)
# as an `add_con!` augmentation. The generator's filter `inner == outer` maps each inner
# element to the (outer) constraint row it contributes to — i.e. `row(par) => term(par)`,
# the direct analogue of `add_con!(c9, a.bus => p[a.i] for a in data.arc)`.
function _augment_function_constraint!(
    c,
    cons,
    pos::Bool,
    gen::FilteredSumGenerator,
    outer_iterators,
    var_to_idx,
)
    @assert length(gen.iterators) == 1 "Only single-iterator generators are supported"
    @assert gen.filter.head == :(==) "Only equality (`==`) filters are supported"
    expr, inner_pars = _exagen(gen.func, gen.iterators, var_to_idx)
    a1, a2 = gen.filter.args
    inner, outer = if a1 isa IteratorValues && a1.iterators === gen.iterators
        a1, a2
    elseif a2 isa IteratorValues && a2.iterators === gen.iterators
        a2, a1
    else
        error("Unsupported generator filter: neither side is the inner iterator")
    end
    @assert outer isa IteratorValues && outer.iterators === outer_iterators "filter must compare the inner iterator to the outer (constraint) iterator"
    # Map each outer (constraint) row key to its 1-based row within `cons`.
    outer_vals = outer_iterators[1].values
    row_of = Dict(outer_vals[k][outer.value_index] => k for k in eachindex(outer_vals))
    # Append the target row to each inner parameter tuple, so the augmentation is `row => term`.
    w = length(first(inner_pars))
    aug_pars = [(par..., row_of[par[inner.value_index]]) for par in inner_pars]
    row = ExaModels.DataIndexed(ExaModels.DataSource(), w + 1)
    signed = pos ? expr : -expr
    c, _ = ExaModels.add_con!(c, cons, (row => signed for _ in aug_pars))
    return c
end

# Convert GenOpt expression trees to ExaModels format

exagen(α::Number, _, _) = α

function exagen(f::MOI.ScalarNonlinearFunction, offsets, var_to_idx)
    if f.head == :getindex
        v = f.args[1]
        if v isa ContiguousArrayOfVariables
            idx = exagen(f.args[2], offsets, var_to_idx)
            # Translate MOI-space offset to ExaModels-space offset using var_to_idx
            first_moi_vi = MOI.VariableIndex(v.offset + 1)
            exa_offset = var_to_idx[first_moi_vi].idx - 1
            if !iszero(exa_offset)
                idx = exa_offset + idx
            end
            cp = cumprod(v.size)
            for i in 3:length(f.args)
                idx += cp[i - 2] * (exagen(f.args[i], offsets, var_to_idx) - 1)
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
        return ExaModels.op(f.head)((exagen(e, offsets, var_to_idx) for e in f.args)...)
    end
end

function _exagen(func::MOI.ScalarNonlinearFunction, iterators, var_to_idx)
    lengths = map(it -> length(first(it.values)), iterators)
    if length(lengths) == 1 && lengths[] == 1
        cs = nothing
        pars = only.(iterators[].values)
    else
        cs = [0; cumsum(lengths)[1:(end - 1)]]
        pars = vec(
            map(Base.Iterators.ProductIterator(ntuple(i -> iterators[i].values, length(iterators)))) do I
                reduce((i, j) -> tuple(i..., j...), I)
            end
        )
    end
    expr = exagen(func, cs, var_to_idx)
    return expr, pars
end

# Bound helpers for vector sets used by FunctionGenerator constraints
_lower_bounds(::Union{MOI.Zeros, MOI.Nonnegatives}, T) = zero(T)
_lower_bounds(::MOI.Nonpositives, T) = typemin(T)
_upper_bounds(::Union{MOI.Zeros, MOI.Nonpositives}, T) = zero(T)
_upper_bounds(::MOI.Nonnegatives, T) = typemax(T)

end # module
