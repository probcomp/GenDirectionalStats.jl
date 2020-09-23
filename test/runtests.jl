using Gen
using Gen3DRotations
using Test
using LinearAlgebra: norm
import Rotations
using SpecialFunctions: besseli

#@testset "Rotation3D" begin
#
    ## test Rotation3D isapprox
    #v = randn(4)
    #v = v / norm(4)
    #a = Rotation3D(Geometry.UnitQuaternion(v[1], v[2], v[3], v[4]))
    #b = Rotation3D(Geometry.UnitQuaternion(-v[1], -v[2], -v[3], -v[4]))
    #@test isapprox(a, b)
#
    #rot = Rotation3D(Geometry.UnitQuaternion(v[1], v[2], v[3], v[4]))
    #@test isapprox(from_quaternion(to_quaternion(rot)), rot)
#
#end

@testset "hopf fibration" begin

    # identity rotation 

    (direction, rot2) = to_direction_and_plane_rotation(Rot3(1.0, 0.0, 0.0, 0.0))
    @test isapprox(Rotations.rotation_angle(rot2), 0.0, atol=1e-5)
    @test isapprox(direction, UnitVector3(0.0, 0.0, 1.0))

    rot3 = from_direction_and_plane_rotation(UnitVector3(0.0, 0.0, 1.0), Rotations.RotMatrix{2,Float64}(0.0))
    @test isapprox(rot3, Rot3(1.0, 0.0, 0.0, 0.0))

    # check that mapping between (UnitVector3, PlaneRotation) parametrization
    # and Rotation3D is an involution

    for i in 1:20
        rot = uniform_rot3()
        (direction, plane_rotation) = to_direction_and_plane_rotation(rot)
        new_rot = from_direction_and_plane_rotation(direction, plane_rotation)
        @test isapprox(new_rot, rot)
    end
    
   for i in 1:20
        direction = uniform_3d_direction()
        plane_rotation = uniform_rot2()
        rot = from_direction_and_plane_rotation(direction, plane_rotation)
        (new_direction, new_plane_rotation) = to_direction_and_plane_rotation(rot)
        @test isapprox(new_direction, direction)
        @test isapprox(new_plane_rotation, plane_rotation)
    end

end


function rand_axis()
    x, y, z = rand(3)
    axis = [x, y, z]
    axis / norm(axis)
end

@gen function foo()
    # uniform distribution, so all moves should be accepted
    rot ~ uniform_rot3()
end

@testset "uniform_angle_fixed_axis_mh" begin

    # smoke test (these should all accept)
    trace = simulate(foo, ())
    for i=1:10
        @test uniform_angle_fixed_axis_mh(trace, :rot, rand_axis();
            check=true, egocentric=false)[2]
        @test uniform_angle_fixed_axis_mh(trace, :rot, rand_axis();
            check=true, egocentric=true)[2]
    end
end

@testset "flip_around_fixed_axis_mh" begin

    # smoke test
    for i=1:10
        trace = simulate(foo, ())
        trace, acc = flip_around_fixed_axis_mh(trace, :rot, rand_axis(), check=true)
        @test acc
    end

    # actual test (since this move is deterministic)
    # test a flip around the y-axis
    rot = Rot3(Rotations.AngleAxis(pi/4, 1.0, 0.0, 0.0))
    trace, = generate(foo, (), choicemap((:rot, rot)))
    expected = Rot3(
        [-1.0 0.0 0.0;
        0.0 cos(pi/4) sin(pi/4);
        0.0 sin(pi/4) -cos(pi/4)])

    # allocentric
    axis = [0., cos(pi/4), sin(pi/4)]
    new_tr, = flip_around_fixed_axis_mh(trace, :rot, axis; check=true, egocentric=false)
    @test isapprox(new_tr[:rot], expected)
    
    # egocentric
    axis = [0., 1., 0.]
    new_tr, = flip_around_fixed_axis_mh(trace, :rot, axis; check=true, egocentric=true)
    @test isapprox(new_tr[:rot], expected)
end

@testset "small_angle_fixed_axis_mh" begin

    # smoke test (these should all accept)
    trace = simulate(foo, ())
    for i=1:10
        @test small_angle_fixed_axis_mh(trace, :rot, rand_axis(), 0.1;
            check=true, egocentric=false)[2]
        @test small_angle_fixed_axis_mh(trace, :rot, rand_axis(), 0.1;
            check=true, egocentric=true)[2]
    end

end

@testset "small_angle_random_axis_mh" begin

    # smoke test (these should all accept)
    trace = simulate(foo, ())
    for i=1:10
        @test small_angle_random_axis_mh(trace, :rot, 0.1;
            check=true, egocentric=false)[2]
        @test small_angle_random_axis_mh(trace, :rot, 0.1;
            check=true, egocentric=true)[2]
    end
end

@testset "log besseli 1" begin

    actual = Gen3DRotations.log_besseli(1, 300.0; n_terms=1000)
    expected = log(besseli(1, 300.0))
    @test isapprox(actual, expected)

    function test_logbesseli_2(x)
        approx = Inf
        prev_approx = -Inf
        n_terms = 1
    
        while abs(approx - prev_approx) > 0.01
            prev_approx = approx
            approx = Gen3DRotations._log_besseli(1, x; n_terms=n_terms)
            n_terms += 1
        end
        return n_terms
    end
    
    x = 10000
    n_terms = test_logbesseli_2(x)
    @show n_terms
    @show n_terms / x
    @time Gen3DRotations.log_besseli(1, 10)
end
