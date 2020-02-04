using Gen
using Gen3DRotations
using Test
using LinearAlgebra: norm
import Geometry

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

    # smoke test (these should all accept)
    trace = simulate(foo, ())
    for i=1:10
        @test uniform_angle_fixed_axis_mh(trace, :rot, rand_axis();
            check_round_trip=true, egocentric=false)[2]
        @test uniform_angle_fixed_axis_mh(trace, :rot, rand_axis();
            check_round_trip=true, egocentric=true)[2]
    end
end

@testset "flip_around_fixed_axis_mh" begin

    # smoke test
    for i=1:10
        trace = simulate(foo, ())
        trace, acc = flip_around_fixed_axis_mh(trace, :rot, rand_axis(), check_round_trip=true)
        @test acc
    end

    # actual test (since this move is deterministic)
    # test a flip around the y-axis
    q = Geometry.axis_angle_to_quat([1., 0., 0.], pi/4)
    trace, = generate(foo, (), choicemap((:rot, q)))
    expected = Geometry.mat2quat([
        -1   0   0;
        0   cos(pi/4)   sin(pi/4);
        0   sin(pi/4)   -cos(pi/4)])

    # allocentric
    axis = [0., cos(pi/4), sin(pi/4)]
    new_tr, = flip_around_fixed_axis_mh(trace, :rot, axis; check_round_trip=true, egocentric=false)
    @test isapprox(new_tr[:rot], expected)
    

    # egocentric
    axis = [0., 1., 0.]
    new_tr, = flip_around_fixed_axis_mh(trace, :rot, axis; check_round_trip=true, egocentric=true)
    @test isapprox(new_tr[:rot], expected)
end

@testset "small_angle_fixed_axis_mh" begin

    # smoke test (these should all accept)
    trace = simulate(foo, ())
    for i=1:10
        @test small_angle_fixed_axis_mh(trace, :rot, rand_axis(), 0.1;
            check_round_trip=true, egocentric=false)[2]
        @test small_angle_fixed_axis_mh(trace, :rot, rand_axis(), 0.1;
            check_round_trip=true, egocentric=true)[2]
    end

end

@testset "small_angle_random_axis_mh" begin

    # smoke test (these should all accept)
    trace = simulate(foo, ())
    for i=1:10
        @test small_angle_random_axis_mh(trace, :rot, 0.1;
            check_round_trip=true, egocentric=false)[2]
        @test small_angle_random_axis_mh(trace, :rot, 0.1;
            check_round_trip=true, egocentric=true)[2]
    end
end

