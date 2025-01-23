# This file contains code that was formerly part of Julia. License is MIT: http://julialang.org/license

using Enzyme, QuadGK, Test

f1(x) = quadgk(cos, 0.0, x)[1]
f2(x) = quadgk(cos, x, 1.0)[1]
f3(x) = quadgk(y -> cos(x * y), 0.0, 1.0)[1]
f4(xarr) = quadgk(y -> cos(xarr[1] * y) + sin(xarr[2] * y), 0.0, 1.0)[1]
f5(x1, x2arr) = quadgk(y -> cos(x1 * y) + sin(x2arr[1] * y), 0.0, 1.0)[1]

f1_count(x) = quadgk_count(cos, 0.0, x)[1]
f2_count(x) = quadgk_count(cos, x, 1.0)[1]
f3_count(x) = quadgk_count(y -> cos(x * y), 0.0, 1.0)[1]
f4_count(xarr) = quadgk_count(y -> cos(xarr[1] * y) + sin(xarr[2] * y), 0.0, 1.0)[1]
f5_count(x1, x2arr) = quadgk_count(y -> cos(x1 * y) + sin(x2arr[1] * y), 0.0, 1.0)[1]

f1_vec(x) = quadgk(y -> [cos(y)], 0.0, x)[1][1]
f2_vec(x) = quadgk(y -> [cos(y)], x, 1.0)[1][1]
f3_vec(x) = quadgk(y -> [cos(x * y)], 0.0, 1.0)[1][1]
f4_vec(xarr) = sum(quadgk(y -> [cos(xarr[1] * y), cos(xarr[2] * y)], 0.0, 1.0)[1])
f5_vec(x1, x2arr) = sum(quadgk(y -> [cos(x1 * y), sin(x2arr[1] * y)], 0.0, 1.0)[1])

f1_vec_count(x) = quadgk_count(y -> [cos(y)], 0.0, x)[1][1]
f2_vec_count(x) = quadgk_count(y -> [cos(y)], x, 1.0)[1][1]
f3_vec_count(x) = quadgk_count(y -> [cos(x * y)], 0.0, 1.0)[1][1]
f4_vec_count(xarr) = sum(quadgk_count(y -> [cos(xarr[1] * y), sin(xarr[2] * y)], 0.0, 1.0)[1])
f5_vec_count(x1, x2arr) = sum(quadgk_count(y -> [cos(x1 * y), sin(x2arr[1] * y)], 0.0, 1.0)[1])

@testset "Enzyme" begin
    x1, x2 = 0.3, 0.7
    xarr, x2arr = [x1, x2], [x2]
    dI_dx1 = (x1 * cos(x1) - sin(x1)) / x1^2
    dI_dx2 = (x2 * sin(x2) + cos(x2) - 1) / x2^2
    @testset "$label" for (label, broken, _f1, _f2, _f3, _f4, _f5) in [
        ("quadgk", false, f1, f2, f3, f4, f5),
        ("quadgk_count", false, f1_count, f2_count, f3_count, f4_count, f5_count),
        # TODO: custom rule with mixed vector returns not yet supported x/ref https://github.com/EnzymeAD/Enzyme.jl/issues/1692
        ("vec quadgk", true, f1_vec, f2_vec, f3_vec, f4_vec, f5_vec),
        ("vec quadgk_count", true, f1_vec_count, f2_vec_count, f3_vec_count, f4_vec_count, f5_vec_count),
    ]
        let dxarr=make_zero(xarr), dx2arr=make_zero(x2arr)
            @test cos(x1) ≈ Enzyme.autodiff(Reverse, _f1, Active(x1))[1][1]   broken=broken
            @test -cos(x1) ≈ Enzyme.autodiff(Reverse, _f2, Active(x1))[1][1]  broken=broken
            @test dI_dx1 ≈ Enzyme.autodiff(Reverse, _f3, Active(x1))[1][1]    broken=broken
            @test begin
                Enzyme.autodiff(Reverse, _f4, Duplicated(xarr, dxarr))
                [dI_dx1, dI_dx2] ≈ dxarr
            end                                                               broken=broken
            @test begin
                dupx2 = Duplicated(x2arr, dx2arr)
                dx1 = Enzyme.autodiff(Reverse, _f5, Active(x1), dupx2)[1][1]
                [dI_dx1, dI_dx2] ≈ [dx1, dx2arr[1]]
            end                                                               broken=broken
        end
    end
end
