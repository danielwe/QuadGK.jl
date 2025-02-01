module QuadGKEnzymeExt

using Enzyme, LinearAlgebra
using QuadGK: do_quadgk, ReturnSegbuf, InplaceIntegrand
using .EnzymeRules: RevConfig, needs_primal, needs_shadow, AugmentedReturn, VectorSpace

function EnzymeRules.augmented_primal(
    config::RevConfig,
    ::Const{typeof(do_quadgk)},
    ::Type{RT},
    f::F,
    segs::Union{Const{NTuple{N,T}},Active{NTuple{N,T}}},
    n::Const,
    atol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    rtol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    maxevals::Const,
    nrm::Const,
    segbuf::Const,
    eval_segbuf::Const,
) where {RT<:Union{Active,MixedDuplicated,BatchMixedDuplicated},F,N,T}
    fv, segsv, othersv..., segbufv, eval_segbufv = map(
        x -> x.val, (f, segs, n, atol, rtol, maxevals, nrm, segbuf, eval_segbuf)
    )

    # whenever we call do_quadgk we hold on to the primal so we can use it for constructing
    # a shadow if needed, regardless of needs_primal(config)
    if (segbufv isa ReturnSegbuf) && (!(f isa Const) || needs_primal(config) || needs_shadow(config))
        # someone could AD quadgk_segbuf; in this case, we must call do_quadgk if:
        # - !(f isa Const)---we need eval_segbuf_new
        # - needs_primal(config)---as always
        # - needs_shadow(config)---only reliable way to obtain a source for the eval_segbuf part of the shadow
        primal_true = do_quadgk(fv, segsv, othersv..., segbufv, eval_segbufv)
        eval_segbuf_new = last(primal_true)
    elseif !(f isa Const)  # call do_quadgk because we need eval_segbuf_new
        @assert !(segbufv isa ReturnSegbuf)  # sanity check, should compile away
        primal_true..., eval_segbuf_new = do_quadgk(
            fv, segsv, othersv..., ReturnSegbuf(segbufv), eval_segbufv
        )
    elseif needs_primal(config)  # call do_quadgk because we need the primal
        @assert !(segbufv isa ReturnSegbuf)  # sanity check, should compile away
        primal_true = do_quadgk(fv, segsv, othersv..., segbufv, eval_segbufv)
        eval_segbuf_new = nothing
    else  # don't call do_quadgk; no need for either the primal or eval_segbuf_new
        primal_true, eval_segbuf_new = nothing, nothing
    end

    primal = needs_primal(config) ? primal_true : nothing

    shadow = if !needs_shadow(config)
        nothing
    else
        # source such that shadow will comprise one or more copies of make_zero(source)
        source = if isnothing(primal_true)
            a, b = segsv[1], segsv[2]
            Imock = f.val((a + b) / 2) * (b - a)
            Emock = nrm.val(Imock)
            if segbufv isa ReturnSegbuf
                @assert !isnothing(eval_segbuf_new)  # sanity check, should compile away
                (Imock, Emock, eval_segbuf_new)
            else
                (Imock, Emock)
            end
        else
            primal_true
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

    tape1 = if f isa Const
        nothing
    else
        @assert !isnothing(eval_segbuf_new)  # sanity check, should compile away
        eval_segbuf_new
    end
    tape2 = (RT <: Active) ? nothing : shadow
    tape = (tape1, tape2)

    return AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    ::RevConfig,
    ::Const{typeof(do_quadgk)},
    shadow::Active,
    tape,
    f::Annotation,
    segs::Union{Const{NTuple{N,T}},Active{NTuple{N,T}}},
    n::Const,
    atol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    rtol::Union{Const,Active},  # never actually active, but may be inferred as such by runtime activity
    maxevals::Const,
    nrm::Const,
    segbuf::Const,
    eval_segbuf::Const,
) where {N,T}
    dI = first(shadow.val)
    mode = ReverseSplitNoPrimal
    fwd, rev = autodiff_thunk(mode, Const{typeof(call)}, Active, typeof(f), Const{T})
    segsv, nv = (segs.val, n.val)
    atol_r = nothing  # don't use primal atol, it's dimensionally incorrect for gradient
    segbuf_r = nothing
    eval_segbuf_r = first(tape)
    # TODO: Need to decide about accurate and adaptive vs. discretized and fast gradient. Use preferences and let the user decide?
    # A: accurate
    # rtol_r = rtol.val
    # maxevals_r = maxevals.val
    # nrm_r = nrm.val
    # B: fast
    rtol_r = nothing
    maxevals_r = 0
    zI = zero(real(dI))
    nrm_r(@nospecialize(x)) = zI
    df = reverse_inner!!(
        f, fwd, rev, dI, segsv, nv, atol_r, rtol_r, maxevals_r, nrm_r, segbuf_r, eval_segbuf_r
    )
    dsegs = if segs isa Const
        nothing
    else
        # XXX: This evaluates f at endpoints, which quadgk promises not to do
        seg1, segN = segsv[1], segsv[N]
        dseg1 = -LinearAlgebra.dot(f.val(seg1), dI)
        dsegN = LinearAlgebra.dot(f.val(segN), dI)
        zdseg = zero(dseg1)
        (dseg1, ntuple(_ -> zdseg, Val(N - 2))..., dsegN)
    end
    datol = (atol isa Const) ? nothing : zero(dI / atol.val)
    drtol = (rtol isa Const) ? nothing : zero(dI / rtol.val)
    return (df, dsegs, nothing, datol, drtol, nothing, nothing, nothing, nothing)
end

call(f::F, x::T) where {F,T} = f(x)

reverse_inner!!(::Const, @nospecialize(args...)) = nothing

function reverse_inner!!(f::Active, fwd, rev, dI, segs, args::Vararg{Any,M}) where {M}
    df_VS, _ = do_quadgk(segs, args...) do x
        tape, _, _ = fwd(Const(call), f, Const(x))
        dfx, _ = only(rev(Const(call), f, Const(x), dI, tape))
        return VectorSpace(dfx; runtime_inactive=Val(false))
    end
    return df_VS.val
end

function reverse_inner!!(
    (; val, dval)::Union{Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed},
    fwd, rev, dI, segsv::NTuple{N,T}, args::Vararg{Any,M},
) where {N,T,M}
    function integrand!(dfx_VS, x)
        dfx = dfx_VS.val
        make_zero!(dfx; runtime_inactive=Val(false))
        f = Duplicated(val, dfx)
        tape, _, _ = fwd(Const(call), f, Const(x))
        rev(Const(call), f, Const(x), dI, tape)
        return nothing
    end
    df_VS = VectorSpace(dval; runtime_inactive=Val(false))
    d_df_VS = zero(df_VS)
    integrand_x = d_df_VS / oneunit(T) # pre-allocate array of correct type for integrand evaluations
    do_quadgk(InplaceIntegrand(integrand!, d_df_VS, integrand_x), segsv, args...)
    df_VS .+= d_df_VS
    return nothing
end

function reverse_inner!!(
    (; val, dval)::Union{MixedDuplicated,BatchMixedDuplicated},
    fwd, rev, dI, segs, args::Vararg{Any,M},
) where {M}
    dfx = Ref(dval[])  # NOTE: dfx wouldn't be thread safe, but quadgk itself is single-threaded so it's OK
    d_df_VS, _ = do_quadgk(segs, args...) do x
        dfx[] = make_zero(dfx[]; runtime_inactive=Val(false))  # zero out-of-place to avoid return value aliasing
        f = MixedDuplicated(val, dfx)
        tape, _, _ = fwd(Const(call), f, Const(x))
        rev(Const(call), f, Const(x), dI, tape)
        return VectorSpace(dfx[]; runtime_inactive=Val(false))
    end
    # NOTE: We take care to update all the mutable parts of dval in-place instead of only
    # updating the outer Ref through something like dval[] = VectorSpace(dval[]) + d_df_VS.
    # That way, if a user holds their own reference to a part of the shadow, it will be
    # updated like they may expect. (I suppose this is not an acutal API guarantee, but
    # Hyrum's law...)
    dfx[] = d_df_VS.val  # reduce, reuse, recycle
    dval_VS = VectorSpace(dval; runtime_inactive=Val(false))
    dval_VS .+= VectorSpace(dfx; runtime_inactive=Val(false))
    return nothing
end

end # module
