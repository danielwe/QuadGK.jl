module QuadGKEnzymeExt

using QuadGK, Enzyme, LinearAlgebra
using .EnzymeRules: needs_primal, needs_shadow, AugmentedReturn, VectorSpace

function EnzymeRules.augmented_primal(
    config::C, ::Const{typeof(quadgk)}, ::Type{RT}, f::F, segs::Vararg{Annotation{T},N}; kws...
) where {C,RT,F,T,N}
    segvals = ntuple(i -> (@inline; segs[i].val), Val(N))

    # res may hold the primal even if needs_primal(config) == false, for constructing shadow
    primal, eval_segbuf = if f isa Const
        res = if needs_primal(config)
            quadgk(f.val, segvals...; kws...)
        else
            nothing
        end
        res, nothing
    else
        res..., eval_segbuf = quadgk_segbuf(f.val, segvals...; kws...)
        if needs_primal(config)
            res, eval_segbuf
        else
            nothing, eval_segbuf
        end
    end

    shadow = if !needs_shadow(config)
        nothing
    else
        # source such that shadow will comprise one or more copies of make_zero(source)
        source = if isnothing(res)
            a, b = segvals[1], segvals[2]
            f.val((a + b) / 2) * (b - a)
        else
            first(res)  # I
        end
        W = EnzymeRules.width(config)
        if W == 1
            make_zero(source; runtime_inactive=Val(false))
        else
            ntuple(Val(W)) do _
                @inline
                make_zero(source; runtime_inactive=Val(false))
            end
        end
    end

    tape = if RT <: Active
        (eval_segbuf, nothing)
    else
        (eval_segbuf, shadow)
    end

    return AugmentedReturn(primal, shadow, tape)
end

call(f::F, x::T) where {F,T} = f(x)

reverse_inner!!(::Const, @nospecialize(args...); @nospecialize(kws...)) = nothing

function reverse_inner!!(f::Active, dI, segvals, fwd, rev; kws...)
    df_VS, _ = quadgk(segvals...; kws...) do x
        tape, _, _ = fwd(Const(call), f, Const(x))
        dfx, _ = only(rev(Const(call), f, Const(x), dI, tape))
        return VectorSpace(dfx; runtime_inactive=Val(false))
    end
    return df_VS.val
end

function reverse_inner!!((; val, dval)::Duplicated, dI, segvals, fwd, rev; kws...)
    df_VS = VectorSpace(dval; runtime_inactive=Val(false))
    d_df_VS = similar(df_VS)
    quadgk!(d_df_VS, segvals...; kws...) do dfx_VS, x
        dfx = dfx_VS.val
        make_zero!(dfx; runtime_inactive=Val(false))
        f = Duplicated(val, dfx)
        tape, _, _ = fwd(Const(call), f, Const(x))
        rev(Const(call), f, Const(x), dI, tape)
        return nothing
    end
    df_VS .+= d_df_VS
    return nothing
end

function reverse_inner!!((; val, dval)::MixedDuplicated, dI, segvals, fwd, rev; kws...)
    dfx = Ref(dval[])  # XXX: dfx wouldn't be thread safe, but quadgk isn't threaded so it's OK
    d_df_VS, _ = quadgk(segvals...; kws...) do x
        dfx[] = make_zero(dfx[]; runtime_inactive=Val(false))  # NOTE: zero out-of-place to avoid return value aliasing
        f = MixedDuplicated(val, dfx)
        tape, _, _ = fwd(Const(call), f, Const(x))
        rev(Const(call), f, Const(x), dI, tape)
        return VectorSpace(dfx[]; runtime_inactive=Val(false))
    end
    # NOTE: We take care to update all the mutable parts of dval in-place instead of only
    # updating the outer Ref through something like dval[] = VectorSpace(dval[]) + d_df_VS.
    # That way, if a user holds their own reference to a part of the shadow, it will be
    # updated like they may expect.
    dfx[] = d_df_VS.val  # reduce, reuse, recycle
    dval_VS = VectorSpace(dval; runtime_inactive=Val(false))
    dval_VS .+= VectorSpace(dfx; runtime_inactive=Val(false))
    return nothing
end

struct ReturnsZero{T} end
(::ReturnsZero{T})(@nospecialize(x)) where {T} = zero(T)

function EnzymeRules.reverse(
    config::C,
    ::Const{typeof(quadgk)},
    shadow::Active,
    tape::P,
    f::Union{Const,Active,Duplicated,MixedDuplicated},
    segs::Vararg{Annotation{T},N};
    segbuf=nothing,       # swallow as this wouldn't have the right eltype
    eval_segbuf=nothing,  # swallow so we're not relying on kw precedence for replacement
    norm=nothing,
    maxevals=nothing,
    kws...,
) where {C,P,T,N}
    dI = first(shadow.val)
    segvals = ntuple(i -> (@inline; segs[i].val), Val(N))
    thunk = autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(call)}, Active, typeof(f), Const{T})
    eval_segbuf = first(tape)
    norm = ReturnsZero{real(typeof(dI))}()
    df = reverse_inner!!(f, dI, segvals, thunk...; kws..., eval_segbuf, norm, maxevals=0)
    dseg1 = (segs[1] isa Const) ? nothing : -LinearAlgebra.dot(f.val(segvals[1]), dI)
    dsegN = (segs[N] isa Const) ? nothing : LinearAlgebra.dot(f.val(segvals[N]), dI)
    return (df, dseg1, ntuple(_ -> (@inline; nothing), Val(N - 2))..., dsegN)
end

# function Enzyme.EnzymeRules.reverse(config, ofunc::Const{typeof(quadgk)}, dres::Type{<:Union{Duplicated, BatchDuplicated}}, cache, f::Union{Const, Active}, segs::Annotation{T}...; kws...) where {T}
#     dres = cache[2]
#     df = if f isa Const
#         nothing
#     else
#         segbuf = cache[1]
#         fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(call)}, Active, typeof(f), Const{T})
#         _df, _ = quadgk(map(x->x.val, segs)...; kws..., eval_segbuf=segbuf, maxevals=0, norm=f->0) do x
#             tape, prim, shad = fwd(Const(call), f, Const(x))
#             shad .= dres
#             drev = rev(Const(call), f, Const(x), tape)
#             return ClosureVector(drev[1][1])
#         end
#         _df.f
#     end
#     dsegs1 = segs[1] isa Const ? nothing : -LinearAlgebra.dot(f.val(segs[1].val), dres)
#     dsegsn = segs[end] isa Const ? nothing : LinearAlgebra.dot(f.val(segs[end].val), dres)
#     Enzyme.make_zero!(dres)
#     return (df, # f
#             dsegs1,
#             ntuple(i -> nothing, Val(length(segs)-2))...,
#             dsegsn)
# end

end # module
