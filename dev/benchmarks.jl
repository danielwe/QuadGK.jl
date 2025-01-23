using QuadGK, Enzyme, ApproxFun, BenchmarkTools, Profile
using ProfileView: @profview
using ProfileCanvas: @profview_allocs

f4(xarr) = quadgk(y -> cos(xarr[1] * y) + sin(xarr[2] * y), 0.0, 1.0)[1]
f5(x1, x2arr) = quadgk(y -> cos(x1 * y) + sin(x2arr[1] * y), 0.0, 1.0)[1]

let x1 = 0.3, x2 = 0.7, xarr = [x1, x2], x2arr = [x2], dxarr = make_zero(xarr), dx2arr = make_zero(x2arr)
    println("f4(xarr)")

    print("forward: ")
    # f4(xarr)
    # @time f4(xarr)
    @btime f4($xarr)

    autodiff(Reverse, f4, Duplicated(xarr, dxarr))
    print("reverse: ")
    @btime begin
        autodiff(Reverse, f4, Duplicated($xarr, $dxarr))
    end setup = make_zero!($dxarr) evals = 1

    println("f5(x1, x2arr)")

    print("forward: ")
    # f5(x1, x2arr)
    # @time f5(x1, x2arr)
    @btime f5($x1, $x2arr)

    autodiff(Reverse, f5, Active(x1), Duplicated(x2arr, dx2arr))
    print("reverse: ")
    @btime begin
        autodiff(Reverse, f5, Active($x1), Duplicated($x2arr, $dx2arr))
    end setup = make_zero!($dx2arr) evals = 1

    # function test(n)
    #     for _ in 1:n
    #         # make_zero!(dxarr)
    #         # autodiff(Reverse, f4, Duplicated(xarr, dxarr))
    #         make_zero!(dx2arr)
    #         autodiff(Reverse, f5, Active(x1), Duplicated(x2arr, dx2arr))
    #     end
    #     return nothing
    # end
    # @profile test(10)
    # @profview test(50000)
end

# let x1 = 0.3, x2 = 0.7, xarr = [x1, x2], x2arr = [x2]
#     # @profview_allocs f4(xarr) sample_rate=1.0
#     @profview_allocs f5(x1, x2arr) sample_rate=1.0
# end

function fcheb(coeffs)
    y = quadgk(Fun(Chebyshev(0.0 .. 1.0), coeffs), 0.0, 1.0)[1]
    return [y, sin(y)]
end

gcheb(coeffs) = sum(x -> x^2, fcheb(coeffs))

let coeffs = ((-1) .^ (0:99) .* exp.(.-(0:99) ./ 2.8)), dcoeffs = make_zero(coeffs)
    println("gcheb(coeffs)")

    print("forward: ")
    # gcheb(coeffs)
    # @time gcheb(coeffs)
    @btime gcheb($coeffs)

    autodiff(Reverse, gcheb, Duplicated(coeffs, dcoeffs))
    print("reverse: ")
    @btime begin
        autodiff(Reverse, gcheb, Duplicated($coeffs, $dcoeffs))
    end setup = make_zero!($dcoeffs) evals = 1

    # function test(n)
    #     for _ in 1:n
    #         make_zero!(dcoeffs)
    #         autodiff(Reverse, gcheb, Duplicated(coeffs, dcoeffs))
    #     end
    #     return nothing
    # end
    # @profile test(10)
    # @profview test(20000)
end

# let coeffs = ((-1) .^ (0:99) .* exp.(.-(0:99) ./ 2.8)), dcoeffs = make_zero(coeffs)
#     @profview_allocs autodiff(Reverse, fcheb, Duplicated(coeffs, dcoeffs)) sample_rate=1.0
# end
