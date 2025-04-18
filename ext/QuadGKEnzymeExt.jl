module QuadGKEnzymeExt

using Enzyme, LinearAlgebra
using QuadGK: do_quadgk, ReturnSegbuf, InplaceIntegrand
using .EnzymeRules: VectorSpace

const runtime_inactive = Val(false)  # XXX: true would be better if it worked and was performant

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(do_quadgk)},
    ::Type{RT},
    f::F,
    segs::Union{Const{S},Active{S}},
    n::Const,
    atol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    rtol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    maxevals::Const,
    nrm::Const,
    segbuf::Const,
    eval_segbuf::Const,
) where {RT<:Union{Const,Active,MixedDuplicated},F,N,T,S<:NTuple{N,T}}
    fv, segsv, othersv..., segbufv, eval_segbufv = map(
        x -> x.val, (f, segs, n, atol, rtol, maxevals, nrm, segbuf, eval_segbuf)
    )

    needs_primal = EnzymeRules.needs_primal(config)
    needs_shadow = EnzymeRules.needs_shadow(config)
    reverse_needs_segbuf = !(f isa Const)

    if (segbufv isa ReturnSegbuf) && (needs_primal || needs_shadow || reverse_needs_segbuf)
        # someone could AD quadgk_segbuf; in this case, we must call do_quadgk if:
        # - !(f isa Const)---we need eval_segbuf_new
        # - needs_primal(config)---as always
        # - needs_shadow(config)---only reliable way to obtain a source for the eval_segbuf part of the shadow
        primal_true = do_quadgk(fv, segsv, othersv..., segbufv, eval_segbufv)
        eval_segbuf_new = last(primal_true)
    elseif reverse_needs_segbuf
        primal_true..., eval_segbuf_new = do_quadgk(
            fv, segsv, othersv..., ReturnSegbuf(segbufv), eval_segbufv
        )
    elseif needs_primal
        primal_true = do_quadgk(fv, segsv, othersv..., segbufv, eval_segbufv)
        eval_segbuf_new = nothing
    else  # forward pass not needed
        primal_true, eval_segbuf_new = nothing, nothing
    end

    primal = EnzymeRules.needs_primal(config) ? primal_true : nothing

    shadow = if !needs_shadow
        nothing
    else
        # source such that shadow will comprise one or more copies of make_zero(source)
        source = if !isnothing(primal_true)
            primal_true
        else  # create mock primal
            a, b = segsv[1], segsv[2]
            Imock = f.val((a + b) / 2) * (b - a)
            Emock = nrm.val(Imock)
            if segbufv isa ReturnSegbuf
                @assert !isnothing(eval_segbuf_new)
                (Imock, Emock, eval_segbuf_new)
            else
                (Imock, Emock)
            end
        end
        make_zero(source; runtime_inactive)
    end

    tape_segbuf = if !reverse_needs_segbuf
        nothing
    else
        @assert !isnothing(eval_segbuf_new)  # sanity check, should compile away
        eval_segbuf_new
    end
    tape_fval = if EnzymeRules.overwritten(config)[2]
        # wrap in VectorSpace before deepcopy to utilize a specialization that calls
        # recursive_map, which is more performant than generic deepcopy
        deepcopy(VectorSpace(fv; runtime_inactive)).val
    else
        nothing
    end
    tape = (tape_segbuf, shadow, tape_fval)

    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

# function EnzymeRules.reverse(
#     ::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(do_quadgk)},
#     ::Type{<:Const},
#     tape,
#     f::Annotation,
#     segs::Union{Const{S},Active{S}},
#     n::Const,
#     atol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
#     rtol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
#     maxevals::Const,
#     nrm::Const,
#     segbuf::Const,
#     eval_segbuf::Const,
# ) where {N,T,S<:NTuple{N,T}}
#     df = if f isa Active
#         make_zero(f.val; runtime_inactve)
#     else
#         if !(f isa Const)
#             make_zero!(f.dval; runtime_inactive)
#         end
#         nothing
#     end
#     dsegs = (segs isa Const) ? nothing : make_zero(segs.val; runtime_inactive)
#     datol = (atol isa Const) ? nothing : make_zero(atol.val; runtime_inactive)
#     drtol = (rtol isa Const) ? nothing : make_zero(rtol.val; runtime_inactive)
#     return (df, dsegs, nothing, datol, drtol, nothing, nothing, nothing, nothing)
# end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(do_quadgk)},
    shadow::Active,
    tape::Tuple{Any,Nothing,Any},
    f::F,
    segs::Union{Const{S},Active{S}},
    n::Const,
    atol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    rtol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    maxevals::Const,
    nrm::Const,
    segbuf::Const,
    eval_segbuf::Const,
) where {F,N,T,S<:NTuple{N,T}}
    dI = first(shadow.val)
    eval_segbuf_r = first(tape)
    # XXX: workaround for Enzyme.jl#2304: would like to use the primal from the tape and
    # accumulate into the shadow from the argument, but they often contain values with
    # mismatched shapes, so this may break
    ff = EnzymeRules.overwritten(config)[2] ? mergeannotation(last(tape), f) : f
    fwd, rev = autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(call)}, Active, F, Const{T})
    segsv, nv = segs.val, n.val
    atol_r = nothing  # don't use primal atol, it's dimensionally incorrect for gradient
    segbuf_r = nothing
    # TODO: need to decide about accurate and adaptive vs. discretized and fast gradient.
    #
    # A: accurate
    # rtol_r = rtol.val
    # maxevals_r = maxevals.val
    # nrm_r = nrm.val
    #
    # B: fast
    rtol_r = nothing
    maxevals_r = 0
    zI = zero(real(dI))
    nrm_r(@nospecialize(x)) = zI
    #
    # TODO: use preferences and let the user decide?
    df = reverse_inner!!(
        ff, fwd, rev, dI, segsv, nv, atol_r, rtol_r, maxevals_r, nrm_r, segbuf_r, eval_segbuf_r
    )
    dsegs = if segs isa Const
        nothing
    else
        # XXX: this evaluates f at endpoints, which quadgk promises not to do---but only if
        # you're differentiating wrt. endpoints, in which case the user asked for it, except
        # they may not have; constant endpoints may be passed as active due to Enzyme
        # activity inference limitations rather than user intent
        fv = ff.val
        seg1, segN = segsv[1], segsv[N]
        dseg1 = -LinearAlgebra.dot(fv(seg1), dI)
        dsegN = LinearAlgebra.dot(fv(segN), dI)
        dseg_zero = zero(dseg1)
        (dseg1, ntuple(_ -> dseg_zero, Val(N - 2))..., dsegN)
    end
    datol = (atol isa Const) ? nothing : make_zero(atol.val; runtime_inactive)
    drtol = (rtol isa Const) ? nothing : make_zero(rtol.val; runtime_inactive)
    return (df, dsegs, nothing, datol, drtol, nothing, nothing, nothing, nothing)
end

mergeannotation(newxval::T, ::Const{T}) where {T} = Const(newxval)::Const{T}
mergeannotation(newxval::T, ::Active{T}) where {T} = Active(newxval)::Active{T}
mergeannotation(newxval::T, x::Duplicated{T}) where {T} = Duplicated(newxval, x.dval)::Duplicated{T}
mergeannotation(newxval::T, x::DuplicatedNoNeed{T}) where {T} = DuplicatedNoNeed(newxval, x.dval)::DuplicatedNoNeed{T}
mergeannotation(newxval::T, x::MixedDuplicated{T}) where {T} = MixedDuplicated(newxval, x.dval)::MixedDuplicated{T}

call(f::F, x::T) where {F,T} = f(x)

reverse_inner!!(::Const, @nospecialize(args...)) = nothing

function reverse_inner!!(f::Active, fwd, rev, dI, segs, args::Vararg{Any,M}) where {M}
    df_VS, _ = do_quadgk(segs, args...) do x
        tape, _, _ = fwd(Const(call), f, Const(x))
        dfx = first(only(rev(Const(call), f, Const(x), dI, tape)))
        return VectorSpace(dfx; runtime_inactive)
    end
    return df_VS.val
end

function reverse_inner!!(
    f::Union{Duplicated,DuplicatedNoNeed}, fwd, rev, dI, segs::NTuple{N,T}, args::Vararg{Any,M}
) where {N,T,M}
    fval = f.val
    df_VS = VectorSpace(f.dval; runtime_inactive)
    d_df_VS = zero(df_VS)
    integrand! = InplaceIntegrand(d_df_VS, d_df_VS / oneunit(T)) do dfx_VS, x
        dfx = dfx_VS.val
        make_zero!(dfx; runtime_inactive)
        fx = Duplicated(fval, dfx)
        tape, _, _ = fwd(Const(call), fx, Const(x))
        rev(Const(call), fx, Const(x), dI, tape)
        return nothing
    end
    do_quadgk(integrand!, segs, args...)
    df_VS .+= d_df_VS
    return nothing
end

function reverse_inner!!(f::MixedDuplicated, fwd, rev, dI, segs, args::Vararg{Any,M}) where {M}
    # NOTE: shared dfx would lead to data races if do_quadgk were multithreaded, but it's not
    fval = f.val
    fx = MixedDuplicated(fval, Ref{typeof(fval)}())
    d_df_VS, _ = do_quadgk(segs, args...) do x
        dfx = fx.dval
        dfx[] = make_zero(fx.val; runtime_inactive)
        tape, _, _ = fwd(Const(call), fx, Const(x))
        rev(Const(call), fx, Const(x), dI, tape)
        return VectorSpace(dfx[]; runtime_inactive)
    end
    # NOTE: need to update all the mutable parts of dval in-place, not just the outer Ref,
    # so f.dval[] = (VectorSpace(f.dval[]) + d_df_VS).val would be incorrect
    d_df = fx.dval  # reduce, reuse, recycle
    d_df[] = d_df_VS.val
    df_VS = VectorSpace(f.dval; runtime_inactive)
    df_VS .+= VectorSpace(d_df; runtime_inactive)
    return nothing
end

end # module
