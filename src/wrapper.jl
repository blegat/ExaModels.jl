# Extension points used by ExaModelsMOI and ExaModelsGenOpt extensions

"""
    copy_extra_constraints!(c, moim, var_to_idx, con_to_idx, T)

Hook for extensions to add extra constraint types after standard MOI constraints
are processed. Default is a no-op, defined in ExaModelsMOI.
"""
function copy_extra_constraints! end

"""
    is_extension_type(::Type{F}) -> Bool

Return `true` if `F` is a function type handled by an extension.
Used by `check_supported` and `supports_constraint` to whitelist extension types.
"""
function is_extension_type end
is_extension_type(::Type) = false

"""
    exafy_extension_obj_arg(m, var_to_idx) -> Union{Nothing, Tuple}

Try to convert an objective function argument `m` to an `(expr, pars)` tuple
for ExaModels. `var_to_idx` maps `MOI.VariableIndex` to `(type, idx)` named tuples.
Returns `nothing` if the type is not handled by any extension.
"""
function exafy_extension_obj_arg end

"""
    copy_extension_con_augmentations!(c, cons, deferred, var_to_idx)

Hook for extensions to apply constraint augmentations (via [`add_con!`](@ref)) for
extension-type terms (e.g. GenOpt `SumGenerator`/`FilteredSumGenerator`) that were
found as additive terms inside `ScalarNonlinearFunction` constraints and deferred by
`copy_constraints!`.

`cons` is the base constraint block returned by `add_con`. `deferred` is a vector of
`(local_row, pos, term)` tuples, where `local_row` is the 1-based row within `cons`,
`pos` is the sign (`true` = add, `false` = subtract) and `term` is the extension
object. Returns the updated core.
"""
function copy_extension_con_augmentations! end

"""
    op(s::Symbol)

Map a Symbol to the corresponding Julia function. Used by both ExaModelsMOI
and ExaModelsGenOpt for expression tree conversion.
"""
function op end
