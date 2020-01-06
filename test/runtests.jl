using Gen
using Gen3DRotations
using Test
using LinearAlgebra: norm

function rand_axis()
    x, y, z = rand(3)
    axis = [x, y, z]
    axis / norm(axis)
end

@gen function foo()
    # uniform distribution, so all moves should be accepted
    @trace(uniform_3d_rotation(), :rot)
end

@testset "uniform_angle_fixed_axis_mh" begin
    trace = simulate(foo, ())
    for i=1:10
        trace, acc = uniform_angle_fixed_axis_mh(trace, :rot, rand_axis(), check_round_trip=true)
        @test acc
    end
end

@testset "flip_around_fixed_axis_mh" begin
    for i=1:10
        trace = simulate(foo, ())
        trace, acc = flip_around_fixed_axis_mh(trace, :rot, rand_axis(), check_round_trip=true)
        @test acc
    end
end

@testset "small_angle_fixed_axis_mh" begin
    trace = simulate(foo, ())
    for i=1:10
        trace, acc = small_angle_fixed_axis_mh(trace, :rot, rand_axis(), 0.1; check_round_trip=true)
        @test acc
    end
end

@testset "small_angle_random_axis_mh" begin
    trace = simulate(foo, ())
    for i=1:10
        trace, acc = small_angle_random_axis_mh(trace, :rot, 0.1; check_round_trip=true)
        @test acc
    end
end

