using Gen: Distribution
import Geometry
import Quaternions
using Geometry: UnitQuaternion
import Gen: logpdf, logpdf_grad, random, has_output_grad, has_argument_grads, logsumexp
import Distributions
using LinearAlgebra: norm, cross, det
using StaticArrays: SVector

########################################
# uniform distribution on 3D rotations #
########################################

struct Uniform3DRotation <: Distribution{UnitQuaternion} end

const uniform_3d_rotation = Uniform3DRotation()

function logpdf(::Uniform3DRotation, r::UnitQuaternion)
    # -log(area of 3 sphere)
    -log(2 * pi * pi)
end

function logpdf_grad(::Uniform3DRotation, r::UnitQuaternion)
    error("Not implemented")
    (nothing,)
end

function random(::Uniform3DRotation)
    w = normal(0, 1)
    x = normal(0, 1)
    y = normal(0, 1)
    z = normal(0, 1)
    n = norm([w, x, y, z])
    UnitQuaternion(w/n, x/n, y/n, z/n)
end

has_output_grad(::Uniform3DRotation) = false

has_argument_grads(::Uniform3DRotation) = ()

(::Uniform3DRotation)() = random(Uniform3DRotation())

export Uniform3DRotation, uniform_3d_rotation

#################################################
# Von Mises Fisher distribution on 3D rotations #
#################################################

# note: this is the Von Mises Fisher distribution on the 3-sphere

struct VonMisesFisher3DRotation <: Distribution{UnitQuaternion} end

const vmf_3d_rotation = VonMisesFisher3DRotation()

function logpdf(::VonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion, k::Float64)
    d = Distributions.VonMisesFisher([mu.w, mu.x, mu.y, mu.z], k)
    log(0.5) + logsumexp([Distributions.logpdf(d, [r.w, r.x, r.y, r.z]), Distributions.logpdf(d, [-r.w, -r.x, -r.y, -r.z])])
end

function logpdf_grad(::VonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion, k::Float64)
    error("Not implemented")
    (nothing,)
end

function random(::VonMisesFisher3DRotation, mu::UnitQuaternion, k::Float64)
    sgn = 1 - 2*bernoulli(0.5)
    d = Distributions.VonMisesFisher(sgn * [mu.w, mu.x, mu.y, mu.z], k)
    v = rand(d)
    UnitQuaternion(v[1], v[2], v[3], v[4])
end

has_output_grad(::VonMisesFisher3DRotation) = false

has_argument_grads(::VonMisesFisher3DRotation) = (false, false)

(::VonMisesFisher3DRotation)(mu, k) = random(VonMisesFisher3DRotation(), mu, k)

export VonMisesFisher3DRotation, vmf_3d_rotation

###########################################
# mixture of VMF and uniform distribution #
###########################################

struct UniformVonMisesFisher3DRotation <: Distribution{UnitQuaternion} end

const uniform_vmf_3d_rotation = UniformVonMisesFisher3DRotation()

function logpdf(
        ::UniformVonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion,
        k::Float64, prob_uniform::Float64)
    lp_vmf = logpdf(vmf_3d_rotation, r, mu, k) + log(1. - prob_uniform)
    lp_uniform = logpdf(uniform_3d_rotation, r) + log(prob_uniform)
    logsumexp(lp_vmf, lp_uniform)
end

function logpdf_grad(::UniformVonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion, k::Float64)
    error("Not implemented")
    (nothing, nothing, nothing, nothing)
end

function random(
        ::UniformVonMisesFisher3DRotation, mu::UnitQuaternion,
        k::Float64, prob_uniform::Float64)
    if bernoulli(prob_uniform)
        uniform_3d_rotation()
    else
        vmf_3d_rotation(mu, k)
    end
end

has_output_grad(::UniformVonMisesFisher3DRotation) = false

has_argument_grads(::UniformVonMisesFisher3DRotation) = (false, false, false, false)

(::UniformVonMisesFisher3DRotation)(mu, k, prob_uniform) = random(UniformVonMisesFisher3DRotation(), mu, k, prob_uniform)

export UniformVonMisesFisher3DRotation, uniform_vmf_3d_rotation

###########################################################################################
# distributions on the sphere S2 (directions in 3D space) and circle S1 (plane rotations) #
###########################################################################################

struct S2
    v::SVector{3,Float64}
end

const Direction3D = S2

function S2(a, b, c)
    v = SVector{3,Float64}(a, b, c)
    @assert isapprox(norm(v), 1.0)
    return S2(v)
end

function Base.rand(::Type{S2})
    v = SVector{3,Float64}(randn(), randn(), randn())
    return S2(v / norm(v))
end

total_area(::Type{S2}) = 4 * pi

struct S1
    v::SVector{2,Float64}
end

const PlaneRotation = S1

function S1(a, b)
    v = SVector{2,Float64}(a, b)
    @assert isapprox(norm(v), 1.0)
    return S1(v)
end

angle_to_plane_rotation(theta::Real) = S1(SVector{2,Float64}(cos(theta), sin(theta)))
plane_rotation_to_angle(x::S1) = atan(x.v[2], x.v[1])

function Base.rand(::Type{S1})
    v = SVector{2,Float64}(randn(), randn())
    return S1(v / norm(v))
end

total_area(::Type{S1}) = 2 * pi

struct UniformOnSphere{S} <: Distribution{S} end

logpdf(::UniformOnSphere{S}, x) where {S} = -log(total_area(S))

function logpdf_grad(::UniformOnSphere, x)
    error("Not implemented")
    return (nothing,)
end

random(::UniformOnSphere{S}) where {S} = rand(S)
has_output_grad(::UniformOnSphere) = false
has_argument_grads(::UniformOnSphere) = ()
(::UniformOnSphere{S})() where {S} = random(UniformOnSphere{S}())

const uniform_3d_direction = UniformOnSphere{S2}()
const uniform_plane_rotation = UniformOnSphere{S1}()

struct VonMisesFisher{S} <: Distribution{S} end

function logpdf(::VonMisesFisher{S}, x::S, mu::S, k::Real) where {S}
    return logpdf(Distributions.VonMisesFisher(mu.v, k), x.v)
end

function logpdf_grad(::VonMisesFisher, x, mu, k)
    error("Not implemented")
    return (nothing,nothing,nothing)
end

function random(::VonMisesFisher{S}) where {S}
    return rand(Distributions.VonMisesFisher(mu.v, k))
end

has_output_grad(::VonMisesFisher) = false
has_argument_grads(::VonMisesFisher) = ()
(::VonMisesFisher{S})() where {S} = random(VonMisesFisher{S}())

const vmf_3d_direction = VonMisesFisher{S2}()
const von_mises_plane_rotation = VonMisesFisher{S1}()

export uniform_3d_direction, uniform_plane_rotation, vmf_3d_direction, von_mises_plane_rotation, Direction3D, PlaneRotation
export angle_to_plane_rotation, plane_rotation_to_angle

#############################
# change of parametrization #
#############################

normalize(vec) = vec / norm(vec)

# Finds the rotation that rotates that minimizes the cosine distance, giving a unique way of defining the new axes
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677
function find_minimal_angle_rotation(a, b)
    a = normalize(a)
    b = normalize(b)
    if isapprox(a, -b)
        @warn "Two vectors are almost opposites; beware"
    end
    v = cross(a, b)
    s = norm(v)
    c = a' * b
    Vcross = [
        0.0     -v[3]       v[2];
        v[3]     0.0       -v[1];
       -v[2]     v[1]       0.0]
    R = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] .+ Vcross .+ ((Vcross * Vcross) / (1 + c))
    @assert isapprox(det(R), 1.0)
    return R
end

function check_angle_bounds(angle)
    if angle < -pi || angle > pi
        error("angle out of bounds [-pi, pi]")
    end
end

function to_rotation_matrix(z_axis, angle_around_z_axis)
    if !isapprox(norm(z_axis), 1.0)
        error("z_axis was not a unit vector")
    end
    check_angle_bounds(angle_around_z_axis)
    R2DoF = find_minimal_angle_rotation([0.0, 0.0, 1.0], z_axis)
    R1DoF = [
        cos(angle_around_z_axis)   -sin(angle_around_z_axis)    0.0;
        sin(angle_around_z_axis)    cos(angle_around_z_axis)    0.0;
        0.0             0.0             1.0
    ]
    @assert isapprox(det(R1DoF), 1.0)
    R = R2DoF * R1DoF
    return R
end

function is_rotation_around_z(R)
    eps = 1e-6
    return (norm(R * [0.0, 0.0, 1.0] .- [0.0, 0.0, 1.0]) < eps &&
        abs(R[3,1]) < eps && 
        abs(R[3,2]) < eps &&
        isapprox(R[2,2], R[1,1]) &&
        isapprox(R[1,2], -R[2,1]))
end

function from_rotation_matrix(R)
    z_axis = R[:,3]
    R2DoF = find_minimal_angle_rotation([0.0, 0.0, 1.0], z_axis)
    R1DoF = R2DoF \ R
    @assert is_rotation_around_z(R1DoF)
    angle_around_z_axis = atan(R1DoF[2,1], R1DoF[1,1])
    return (z_axis, angle_around_z_axis)
end


function unit_quaternion_to_direction_and_plane_rotation(quat::UnitQuaternion)::Tuple{Direction3D,PlaneRotation}
    R = Geometry.quat2mat(quat)
    (_z_axis, _angle_around_z_axis) = from_rotation_matrix(R)
    direction = Direction3D(_z_axis[1], _z_axis[2], _z_axis[3])
    plane_rotation = angle_to_plane_rotation(_angle_around_z_axis)
    return (direction, plane_rotation)
end

function direction_and_plane_rotation_to_unit_quaternion(direction::Direction3D, plane_rotation::PlaneRotation)::UnitQuaternion
    R = to_rotation_matrix(direction.v, plane_rotation_to_angle(plane_rotation))
    return Geometry.mat2quat(R)
end

export unit_quaternion_to_direction_and_plane_rotation
export direction_and_plane_rotation_to_unit_quaternion

function test_transformation()

    function generate_random_rotation()
        axis = rand(3)
        angle = -pi + 2*pi*rand()
        return Quaternions.rotationmatrix(Quaternions.qrotation(axis, angle))
    end
    
    for i in 1:100
        R = generate_random_rotation()
        (z_axis, angle_around_z_axis) = from_rotation_matrix(R)
        new_R = to_rotation_matrix(z_axis, angle_around_z_axis)
        @assert isapprox(new_R, R)
    end
    
    function generate_random_z_axis_and_angle()
        z_axis = normalize(rand(3))
        angle_around_z_axis = -pi + 2*pi*rand()
        return (z_axis, angle_around_z_axis)
    end
    
    for i in 1:100
        (z_axis, angle_around_z_axis) = generate_random_z_axis_and_angle()
        R = to_rotation_matrix(z_axis, angle_around_z_axis)
        (new_z_axis, new_angle_around_z_axis) = from_rotation_matrix(R)
        @assert isapprox(z_axis, new_z_axis)
        @assert isapprox(angle_around_z_axis, new_angle_around_z_axis)
    end

end

test_transformation()
